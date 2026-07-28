import streamlit as st
from PIL import Image
import cv2
import numpy as np
import re
import easyocr

st.set_page_config(page_title="PL1 VIN Scanner & Comparator", layout="centered")

# Hide Streamlit Cloud throttling banner
st.markdown("""
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# Initialize Session State
if 'checksheet_vin' not in st.session_state:
    st.session_state.checksheet_vin = ""
if 'car_vin' not in st.session_state:
    st.session_state.car_vin = ""

def process_vin_image(pil_image):
    """Preprocesses snapshot image and extracts PL1 VIN."""
    # Convert PIL Image to OpenCV Format
    img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. Image Enhancement (Contrast Boost + Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding to isolate black text on white/metal background
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # 2. Run EasyOCR on processed crisp image
    results = reader.readtext(thresh, detail=0)
    raw_text = "".join(results).upper()
    
    # Display raw extraction in UI for debugging
    st.write(f"🔍 **Raw Detected Text:** `{raw_text if raw_text else 'None'}`")

    # Clean characters (alphanumeric only)
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text)

    if not cleaned:
        return None

    # Force correct PL1 prefix if OCR mistook 1 for I, L, or P11/RL1
    if len(cleaned) >= 3:
        prefix = cleaned[:3]
        if prefix in ["PLI", "PLL", "P11", "RL1", "FL1", "PL1"]:
            cleaned = "PL1" + cleaned[3:]

    # Map illegal VIN characters (I -> 1, O/Q -> 0)
    corrected = []
    for char in cleaned:
        if char == 'I':
            corrected.append('1')
        elif char in ['O', 'Q']:
            corrected.append('0')
        else:
            corrected.append(char)
    corrected_str = "".join(corrected)

    # Search for explicit 17-char PL1 string
    pl1_matches = re.findall(r'PL1[A-HJ-NPR-Z0-9]{14}', corrected_str)
    if pl1_matches:
        return pl1_matches[0]

    # Search for generic 17-char VIN string
    vin_matches = re.findall(r'[A-HJ-NPR-Z0-9]{17}', corrected_str)
    if vin_matches:
        return vin_matches[0]

    # Fallback to 10+ characters
    return corrected_str[:17] if len(corrected_str) >= 10 else None

# UI Header
st.title("🚗 VIN Comparison Tool")
st.write("Take a crisp, close-up photo of the VIN sticker or checksheet.")

tabs = st.tabs(["1️⃣ Scan Checksheet", "2️⃣ Scan Car VIN"])

# ----------------- TAB 1: Checksheet -----------------
with tabs[0]:
    st.subheader("Step 1: Checksheet VIN")
    checksheet_file = st.camera_input("Capture Checksheet VIN", key="cam_sheet")
    
    if checksheet_file:
        img = Image.open(checksheet_file)
        with st.spinner("Processing VIN..."):
            extracted = process_vin_image(img)
            if extracted:
                st.session_state.checksheet_vin = extracted
                st.success(f"Captured Checksheet VIN: `{extracted}`")
            else:
                st.error("Could not read a valid VIN. Please retake photo closer to text.")

# ----------------- TAB 2: Car VIN -----------------
with tabs[1]:
    st.subheader("Step 2: Car VIN")
    car_file = st.camera_input("Capture Car VIN (Door Jamb / Plate)", key="cam_car")
    
    if car_file:
        img = Image.open(car_file)
        with st.spinner("Processing VIN..."):
            extracted = process_vin_image(img)
            if extracted:
                st.session_state.car_vin = extracted
                st.success(f"Captured Car VIN: `{extracted}`")
            else:
                st.error("Could not read a valid VIN. Please retake photo closer to text.")

st.divider()

# ----------------- Comparison Section -----------------
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
        st.success(f"✅ MATCH CONFIRMED!\n\n`{chk}`")
    else:
        st.error(f"❌ MISMATCH DETECTED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
