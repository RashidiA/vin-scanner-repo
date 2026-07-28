import queue
import re
import av
import cv2
import easyocr
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="PL1 VIN Scanner & Comparator", layout="centered")

# Hide Streamlit throttling banner if present
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

# 2. Cache thread-safe Queues
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
    st.session_state.last_raw_ocr = "Waiting for video feed..."

def clean_and_extract_vin(image_crop):
    """Upscales crop, maps OCR errors for PL1 VINs, and extracts 17 chars."""
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

    # Upscale 2x for OCR clarity
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Run EasyOCR (allow upper letters + numbers)
    results = reader.readtext(resized, detail=0)
    raw_text = "".join(results).upper()
    
    # Store for live UI debugging
    if debug_queue.empty():
        debug_queue.put(raw_text)

    # Clean to alphanumeric only (KEEP 'I', 'O', 'Q' initially so we can correct them)
    cleaned_alpha = re.sub(r"[^A-Z0-9]", "", raw_text)

    if not cleaned_alpha:
        return None

    # --- SPECIFIC PL1 PREFIX CORRECTION ---
    # Fix common OCR misreads of "PL1" (e.g., PLI, PLL, P11, RL1, FL1)
    if len(cleaned_alpha) >= 3:
        prefix = cleaned_alpha[:3]
        if prefix in ["PLI", "PLL", "P11", "RL1", "FL1", "PL1"]:
            cleaned_alpha = "PL1" + cleaned_alpha[3:]

    # --- STANDARD VIN CHARACTER REPLACEMENT ---
    # Replace illegal VIN characters: I -> 1, O -> 0, Q -> 0
    corrected = []
    for char in cleaned_alpha:
        if char == "I":
            corrected.append("1")
        elif char in ["O", "Q"]:
            corrected.append("0")
        else:
            corrected.append(char)
            
    corrected_str = "".join(corrected)

    # 1. Look for explicit 17-character PL1 VIN pattern
    pl1_matches = re.findall(r"PL1[A-HJ-NPR-Z0-9]{14}", corrected_str)
    if pl1_matches:
        return pl1_matches[0]

    # 2. Look for any 17-character valid VIN pattern
    vin_matches = re.findall(r"[A-HJ-NPR-Z0-9]{17}", corrected_str)
    if vin_matches:
        return vin_matches[0]

    # 3. Fallback: If 10+ chars detected, pass best effort for manual correction
    if len(corrected_str) >= 10:
        return corrected_str[:17]

    return None

st.title("🚗 VIN Comparison Tool")
st.write("Align the **PL1** VIN inside the green bounding box.")

mode = st.radio(
    "Scanning target:", ["1️⃣ Checksheet VIN", "2️⃣ Car VIN"], horizontal=True
)

# Video Callback Processor
class VINVideoProcessor:
    def __init__(self, res_queue):
        self.res_queue = res_queue
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Broad Target ROI Box (85% width, 35% height)
        box_w, box_h = int(w * 0.85), int(h * 0.35)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        x2, y2 = x1 + box_w, y1 + box_h

        # Throttle: Evaluate 1 frame every 10 frames (~0.3s)
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            crop = img[y1:y2, x1:x2].copy()
            if self.res_queue.empty():
                self.res_queue.put(crop)

        # Draw Guidance Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img,
            "ALIGN PL1 VIN HERE",
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="vin-scanner-pl1",
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

# Process queued frame
if not result_queue.empty():
    cropped_frame = result_queue.get()
    detected_vin = clean_and_extract_vin(cropped_frame)

    if detected_vin:
        if mode == "1️⃣ Checksheet VIN":
            st.session_state.checksheet_vin = detected_vin
        else:
            st.session_state.car_vin = detected_vin
        st.rerun()

# Display Live OCR Debug Feed
if not debug_queue.empty():
    st.session_state.last_raw_ocr = debug_queue.get()

st.caption(f"🔍 **Live Camera OCR Feed:** `{st.session_state.last_raw_ocr}`")

st.divider()

# Interactive Form & Manual Overrides
col1, col2 = st.columns(2)
with col1:
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN:", value=st.session_state.checksheet_vin
    ).upper()

with col2:
    st.session_state.car_vin = st.text_input(
        "Car VIN:", value=st.session_state.car_vin
    ).upper()

# Verification Check
if st.session_state.checksheet_vin and st.session_state.car_vin:
    chk = st.session_state.checksheet_vin.strip()
    car = st.session_state.car_vin.strip()

    if chk == car:
        st.balloons()
        st.success(f"✅ MATCH CONFIRMED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
    else:
        st.error(f"❌ MISMATCH DETECTED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
