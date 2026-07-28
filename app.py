import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, WebRtcMode
import av
import cv2
import easyocr
import re
import queue

st.set_page_config(page_title="HD VIN Scanner & Comparator", layout="centered")

# Hide Streamlit throttling banner
st.markdown("""
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# 1. Cache EasyOCR
@st.cache_resource
def load_ocr():
    # Load CPU reader with English
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# Thread-safe Queue to pass cropped frames from WebRTC worker to Streamlit
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

if 'checksheet_vin' not in st.session_state:
    st.session_state.checksheet_vin = ""
if 'car_vin' not in st.session_state:
    st.session_state.car_vin = ""

def clean_and_extract_vin(image_crop):
    """Upscales crop and cleans text for EasyOCR."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize / Upscale 2x so character heights are large enough for OCR
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    # Sharpen and threshold image
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    
    # Run EasyOCR
    results = reader.readtext(blurred, detail=0)
    raw_text = "".join(results).upper()
    
    # Remove unwanted noise (keep only A-Z, 0-9 except I, O, Q)
    cleaned = re.sub(r'[^A-HJ-NPR-Z0-9]', '', raw_text)
    
    # Look for 17-char VIN or longest valid sequence
    vins = re.findall(r'[A-HJ-NPR-Z0-9]{17}', cleaned)
    if vins:
        return vins[0]
    elif len(cleaned) >= 10:
        return cleaned[:17]
    return None

st.title("🚗 VIN Comparison Tool")
st.write("Hold camera steady. Align VIN inside the green box.")

mode = st.radio("Scanning target:", ["1️⃣ Checksheet VIN", "2️⃣ Car VIN"], horizontal=True)

# Video Callback
class VINVideoProcessor:
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Define target ROI Box (80% width, 30% height)
        box_w, box_h = int(w * 0.8), int(h * 0.3)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        x2, y2 = x1 + box_w, y1 + box_h

        # Send 1 frame every 15 frames (~0.5 seconds) to Queue for main thread processing
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            crop = img[y1:y2, x1:x2].copy()
            if self.result_queue.empty():
                self.result_queue.put(crop)

        # Draw Guidance Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, "ALIGN VIN HERE", (x1 + 10, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit WebRTC Streamer with explicit HD constraints
webrtc_ctx = webrtc_streamer(
    key="vin-scanner-hd",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VINVideoProcessor(st.session_state.result_queue),
    media_stream_constraints={
        "video": {
            "facingMode": "environment",
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        },
        "audio": False
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}, playsinline=True)
)

# Read frames pushed to Queue by WebRTC and update UI
if not st.session_state.result_queue.empty():
    cropped_frame = st.session_state.result_queue.get()
    detected_vin = clean_and_extract_vin(cropped_frame)
    
    if detected_vin:
        if mode == "1️⃣ Checksheet VIN":
            st.session_state.checksheet_vin = detected_vin
        else:
            st.session_state.car_vin = detected_vin
        st.rerun()

st.divider()

# Interactive Form
col1, col2 = st.columns(2)
with col1:
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN:", 
        value=st.session_state.checksheet_vin
    ).upper()

with col2:
    st.session_state.car_vin = st.text_input(
        "Car VIN:", 
        value=st.session_state.car_vin
    ).upper()

# Comparison Logic
if st.session_state.checksheet_vin and st.session_state.car_vin:
    chk = st.session_state.checksheet_vin.strip()
    car = st.session_state.car_vin.strip()
    
    if chk == car:
        st.balloons()
        st.success(f"✅ MATCH CONFIRMED!\n\n`{chk}`")
    else:
        st.error(f"❌ MISMATCH!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
