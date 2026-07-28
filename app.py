import queue
import re
import av
import cv2
import easyocr
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="HD VIN Scanner & Comparator", layout="centered")

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


# 2. Cache thread-safe Queue across WebRTC and Streamlit contexts
@st.cache_resource
def get_result_queue():
    return queue.Queue()


result_queue = get_result_queue()

# Initialize session state for tracking detected VINs
if "checksheet_vin" not in st.session_state:
    st.session_state.checksheet_vin = ""
if "car_vin" not in st.session_state:
    st.session_state.car_vin = ""


def clean_and_extract_vin(image_crop):
    """Upscales crop, applies image filtering, and extracts VIN string."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

    # Resize / Upscale 2x so character heights are clear for OCR
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Sharpen and reduce noise
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    # Run EasyOCR on processed ROI
    results = reader.readtext(blurred, detail=0)
    raw_text = "".join(results).upper()

    # Clean text to keep standard VIN characters (A-Z, 0-9 except I, O, Q)
    cleaned = re.sub(r"[^A-HJ-NPR-Z0-9]", "", raw_text)

    # Search for standard 17-character VIN pattern or longest sequence
    vins = re.findall(r"[A-HJ-NPR-Z0-9]{17}", cleaned)
    if vins:
        return vins[0]
    elif len(cleaned) >= 10:
        return cleaned[:17]
    return None


st.title("🚗 VIN Comparison Tool")
st.write("Hold camera steady. Align the VIN inside the green box.")

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

        # Target ROI Box (80% width, 30% height)
        box_w, box_h = int(w * 0.8), int(h * 0.3)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        x2, y2 = x1 + box_w, y1 + box_h

        # Throttle frame evaluation: Send 1 frame every 15 frames (~0.5s)
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            crop = img[y1:y2, x1:x2].copy()
            if self.res_queue.empty():
                self.res_queue.put(crop)

        # Draw Guidance Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img,
            "ALIGN VIN HERE",
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="vin-scanner-hd",
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

# Read queued frames from WebRTC worker thread
if not result_queue.empty():
    cropped_frame = result_queue.get()
    detected_vin = clean_and_extract_vin(cropped_frame)

    if detected_vin:
        if mode == "1️⃣ Checksheet VIN":
            st.session_state.checksheet_vin = detected_vin
        else:
            st.session_state.car_vin = detected_vin
        st.rerun()

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
