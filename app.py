import queue
import re
import av
import cv2
import easyocr
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="PL1 VIN Scanner & Comparator", layout="centered")

# Hide Streamlit throttling banner
st.markdown(
    """
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    </style>
""",
    unsafe_allow_html=True,
)

# 1. Cache EasyOCR Reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr()

# 2. Thread-safe Queues
@st.cache_resource
def get_result_queue():
    return queue.Queue()

@st.cache_resource
def get_debug_queue():
    return queue.Queue()

result_queue = get_result_queue()
debug_queue = get_debug_queue()

# Initialize session state
if "checksheet_vin" not in st.session_state:
    st.session_state.checksheet_vin = ""
if "car_vin" not in st.session_state:
    st.session_state.car_vin = ""
if "last_raw_ocr" not in st.session_state:
    st.session_state.last_raw_ocr = "Waiting for active camera feed..."

def preprocess_and_ocr(image_crop):
    """Applies Otsu binarization and extracts PL1 VIN text."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

    # Upscale 2x for character resolution
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement + Otsu Thresholding (Binarization)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run OCR on thresholded image
    results = reader.readtext(thresh, detail=0)
    raw_text = "".join(results).upper()
    
    # Send raw output to debug log
    if debug_queue.empty():
        debug_queue.put(raw_text if raw_text else "[No text detected in crop box]")

    # Filter alphanumeric
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text)
    if not cleaned:
        return None

    # Handle PL1 OCR confusion (PLI, PLL, P11 -> PL1)
    if len(cleaned) >= 3:
        if cleaned[:3] in ["PLI", "PLL", "P11", "RL1", "FL1"]:
            cleaned = "PL1" + cleaned[3:]

    # Map invalid VIN characters
    corrected = []
    for c in cleaned:
        if c == "I":
            corrected.append("1")
        elif c in ["O", "Q"]:
            corrected.append("0")
        else:
            corrected.append(c)
    corrected_str = "".join(corrected)

    # Search for PL1 17-char match
    pl1_matches = re.findall(r"PL1[A-HJ-NPR-Z0-9]{14}", corrected_str)
    if pl1_matches:
        return pl1_matches[0]

    # Search for standard 17-char VIN
    vin_matches = re.findall(r"[A-HJ-NPR-Z0-9]{17}", corrected_str)
    if vin_matches:
        return vin_matches[0]

    # Fallback to partial capture if length >= 10
    if len(corrected_str) >= 10:
        return corrected_str[:17]

    return None

st.title("🚗 VIN Comparison Tool")
st.write("Align the **PL1** VIN inside the green bounding box.")

mode = st.radio(
    "Target to Scan:", ["1️⃣ Checksheet VIN", "2️⃣ Car VIN"], horizontal=True
)

# Video Callback Processor
class VINVideoProcessor:
    def __init__(self, res_queue):
        self.res_queue = res_queue
        self.frame_count = 0
        self.status = "CAMERA ACTIVE"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Define Target ROI Box (85% width, 30% height)
        box_w, box_h = int(w * 0.85), int(h * 0.3)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        x2, y2 = x1 + box_w, y1 + box_h

        # Send frame to queue every 12 frames (~0.4s)
        self.frame_count += 1
        if self.frame_count % 12 == 0:
            self.status = "READING FRAME..."
            crop = img[y1:y2, x1:x2].copy()
            if self.res_queue.empty():
                self.res_queue.put(crop)
        else:
            self.status = "SCANNING..."

        # Draw Target Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Overlay Live Status Banner on Top of Stream
        cv2.putText(
            img,
            f"STATUS: {self.status}",
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamer
webrtc_ctx = webrtc_streamer(
    key="vin-scanner-status",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VINVideoProcessor(result_queue),
    media_stream_constraints={
        "video": {
            "facingMode": "environment",
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        style={"width": "100%"},
        playsinline=True,
    ),
)

# Handle Queued Crop Frame
if not result_queue.empty():
    cropped_frame = result_queue.get()
    detected_vin = preprocess_and_ocr(cropped_frame)

    if detected_vin:
        if mode == "1️⃣ Checksheet VIN":
            st.session_state.checksheet_vin = detected_vin
        else:
            st.session_state.car_vin = detected_vin
        st.rerun()

# Display Debug Output
if not debug_queue.empty():
    st.session_state.last_raw_ocr = debug_queue.get()

# Status Banner
st.info(f"📡 **Live OCR Raw Reading:** `{st.session_state.last_raw_ocr}`")

st.divider()

# Inputs & Matching Logic
col1, col2 = st.columns(2)
with col1:
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN:", value=st.session_state.checksheet_vin
    ).upper()

with col2:
    st.session_state.car_vin = st.text_input(
        "Car VIN:", value=st.session_state.car_vin
    ).upper()

if st.session_state.checksheet_vin and st.session_state.car_vin:
    chk = st.session_state.checksheet_vin.strip()
    car = st.session_state.car_vin.strip()

    if chk == car:
        st.balloons()
        st.success(f"✅ MATCH CONFIRMED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
    else:
        st.error(f"❌ MISMATCH DETECTED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
