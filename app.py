import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, WebRtcMode
import av
import cv2
import easyocr
import re
import time

st.set_page_config(page_title="High-Speed VIN Comparator", layout="centered")

# Hide Streamlit throttling banner
st.markdown("""
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# 1. Cache EasyOCR model
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# Session State for tracking detected VINs
if 'checksheet_vin' not in st.session_state:
    st.session_state.checksheet_vin = ""
if 'car_vin' not in st.session_state:
    st.session_state.car_vin = ""
if 'last_ocr_time' not in st.session_state:
    st.session_state.last_ocr_time = 0

def preprocess_and_ocr(crop_img):
    """Preprocess ROI crop for max OCR accuracy and clean VIN text."""
    # Convert to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Run EasyOCR on optimized crop
    results = reader.readtext(enhanced, allowlist='0123456789ABCDEFGHJKLMNPRSTUVWXYZ')
    
    all_text = "".join([res[1] for res in results]).upper()
    cleaned = re.sub(r'[^A-HJ-NPR-Z0-9]', '', all_text)
    
    # Look for standard 17-character VIN pattern
    match = re.search(r'[A-HJ-NPR-Z0-9]{17}', cleaned)
    return match.group(0) if match else (cleaned if len(cleaned) >= 10 else None)

st.title("🚗 VIN Comparison Tool")
st.write("Align the VIN inside the green bounding box.")

mode = st.radio("Select Target:", ["1️⃣ Checksheet VIN", "2️⃣ Car VIN"], horizontal=True)

# Video Frame Callback
class VINProcessor:
    def __init__(self):
        self.detected_text = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Define ROI (Center box: 60% width, 20% height)
        box_w, box_h = int(w * 0.7), int(h * 0.25)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        x2, y2 = x1 + box_w, y1 + box_h

        # Throttle OCR to run once every 0.8 seconds to prevent video lag
        current_time = time.time()
        if not hasattr(self, 'last_run') or (current_time - self.last_run) > 0.8:
            self.last_run = current_time
            crop = img[y1:y2, x1:x2]
            res = preprocess_and_ocr(crop)
            if res:
                self.detected_text = res

        # Draw green bounding target box on the UI stream
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "ALIGN VIN HERE", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit WebRTC Streamer
ctx = webrtc_streamer(
    key="vin-scanner",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VINProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}, playsinline=True)
)

# Capture detected text from process stream
if ctx.video_processor:
    latest = ctx.video_processor.detected_text
    if latest:
        if mode == "1️⃣ Checksheet VIN":
            st.session_state.checksheet_vin = latest
        else:
            st.session_state.car_vin = latest

st.divider()

# Inputs & Comparison
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

# Verification Check
if st.session_state.checksheet_vin and st.session_state.car_vin:
    chk = st.session_state.checksheet_vin.strip()
    car = st.session_state.car_vin.strip()
    
    if chk == car:
        st.success(f"✅ MATCH CONFIRMED!\n\n`{chk}`")
    else:
        st.error(f"❌ MISMATCH!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
