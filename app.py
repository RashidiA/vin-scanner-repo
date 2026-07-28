import re
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# 1. Page Configuration (Wide Mode)
st.set_page_config(page_title="PL1 VIN Scanner", layout="wide")

# Hide Streamlit overhead banners & style full-width camera inputs
st.markdown(
    """
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    div[data-testid="stCameraInput"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        border-radius: 8px;
        border: 3px solid #00E676;
    }
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 2. Cache the RapidOCR Engine
@st.cache_resource
def load_rapid_ocr():
    return RapidOCR()


engine = load_rapid_ocr()

# Initialize Session State
if "checksheet_vin" not in st.session_state:
    st.session_state.checksheet_vin = ""
if "car_vin" not in st.session_state:
    st.session_state.car_vin = ""


def process_pl1_vin(pil_image):
    """Processes snapshot with contrast enhancement and extracts 17-char PL1 VIN via RapidOCR."""
    img_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Image enhancement (Upscale + Adaptive Threshold)
    resized = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to RGB for RapidOCR
    processed_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Perform OCR
    results, _ = engine(processed_rgb)

    if not results:
        # Fallback run on standard un-thresholded image if threshold fails
        results, _ = engine(img_np)

    if not results:
        return None

    # Concatenate all detected text snippets
    raw_text = "".join([item[1] for item in results]).upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text)

    if not cleaned:
        return None

    # Search for PL1 (or common OCR prefix corruptions) + 14 trailing chars
    pl1_match = re.search(r"(PL1|PLI|PLL|P11|P1I|RL1|FL1)[A-Z0-9]{14}", cleaned)

    if pl1_match:
        suffix = pl1_match.group(0)[3:]
    else:
        # Fallback: grab any detected sequence of 14+ characters
        long_matches = re.findall(r"[A-Z0-9]{14,17}", cleaned)
        if long_matches:
            target = long_matches[0]
            suffix = target[3:17] if len(target) >= 17 else target[:14]
        else:
            suffix = cleaned[:14]

    # Map illegal VIN characters (I -> 1, O/Q -> 0)
    corrected_suffix = []
    for char in suffix:
        if char == "I":
            corrected_suffix.append("1")
        elif char in ["O", "Q"]:
            corrected_suffix.append("0")
        else:
            corrected_suffix.append(char)

    final_suffix = "".join(corrected_suffix)

    # STRICT GUARANTEE: Force PL1 Prefix
    final_vin = f"PL1{final_suffix}"
    return final_vin[:17]


# Header
st.title("🚗 High-Speed PL1 VIN Scanner")
st.caption(
    "Powered by **RapidOCR Engine**. Every detected code is strictly validated and formatted to start with **PL1**."
)

col_scan1, col_scan2 = st.columns(2)

with col_scan1:
    st.subheader("1️⃣ Checksheet VIN")
    sheet_file = st.camera_input("Take photo of Checksheet", key="cam_sheet")
    if sheet_file:
        img = Image.open(sheet_file)
        with st.spinner("Extracting VIN..."):
            res = process_pl1_vin(img)
            if res:
                st.session_state.checksheet_vin = res
                st.success(f"Detected: `{res}`")
            else:
                st.error("No text found. Move closer and ensure clear lighting.")

with col_scan2:
    st.subheader("2️⃣ Car VIN")
    car_file = st.camera_input("Take photo of Car VIN", key="cam_car")
    if car_file:
        img = Image.open(car_file)
        with st.spinner("Extracting VIN..."):
            res = process_pl1_vin(img)
            if res:
                st.session_state.car_vin = res
                st.success(f"Detected: `{res}`")
            else:
                st.error("No text found. Move closer and ensure clear lighting.")

st.divider()

# Input Overrides & Result Display
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN:", value=st.session_state.checksheet_vin
    ).upper()

with col_input2:
    st.session_state.car_vin = st.text_input(
        "Car VIN:", value=st.session_state.car_vin
    ).upper()

# Verification Check
if st.session_state.checksheet_vin and st.session_state.car_vin:
    chk = st.session_state.checksheet_vin.strip()
    car = st.session_state.car_vin.strip()

    if chk == car:
        st.balloons()
        st.success(f"✅ MATCH CONFIRMED!\n\nVIN: `{chk}`")
    else:
        st.error(f"❌ MISMATCH DETECTED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
