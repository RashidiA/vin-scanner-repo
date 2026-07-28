import streamlit as st
from PIL import Image
import cv2
import numpy as np
import re
import easyocr

# 1. Set Wide Page Layout
st.set_page_config(page_title="PL1 VIN Scanner", layout="wide")

# 2. Inject CSS for Widescreen Camera Viewport & Hide Throttling Banner
st.markdown("""
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    
    /* Make camera input and video containers take full width */
    div[data-testid="stCameraInput"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        border-radius: 10px;
        border: 3px solid #00FF00;
    }
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
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

def extract_strict_pl1_vin(pil_image):
    """Processes image and forcibly extracts/formats a PL1-prefixed 17-char VIN."""
    img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale & Threshold for Contrast
    resized = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # OCR Extraction
    results = reader.readtext(thresh, detail=0)
    raw_text = "".join(results).upper()
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text)

    if not cleaned:
        return None

    # Step A: Search for PL1 (or common OCR corruption: PLI, PLL, P11, RL1) + 14 chars
    pl1_match = re.search(r'(PL1|PLI|PLL|P11|P1I|RL1|FL1)[A-Z0-9]{14}', cleaned)
    
    if pl1_match:
        found_str = pl1_match.group(0)
        suffix = found_str[3:]  # Extract remaining 14 characters
    else:
        # Step B: Hard Fallback - Find any 14 to 17 character string
        long_matches = re.findall(r'[A-Z0-9]{14,17}', cleaned)
        if long_matches:
            target = long_matches[0]
            suffix = target[3:17] if len(target) >= 17 else target[:14]
        else:
            # Step C: Fallback for short/partial OCR reads
            suffix = cleaned[:14]

    # Clean the 14-char suffix (replace invalid VIN chars: I->1, O/Q->0)
    corrected_suffix = []
    for char in suffix:
        if char == 'I':
            corrected_suffix.append('1')
        elif char in ['O', 'Q']:
            corrected_suffix.append('0')
        else:
            corrected_suffix.append(char)
            
    final_suffix = "".join(corrected_suffix)

    # ALWAYS FORCIBLY PREPEND PL1
    final_vin = f"PL1{final_suffix}"
    
    # Trim to exactly 17 characters
    return final_vin[:17]

# Title Banner
st.title("🚗 PL1 VIN Matcher & Inspector")
st.caption("Widescreen camera enabled. All readings are strictly enforced to start with **PL1**.")

# Widescreen Layout Columns
col_scan1, col_scan2 = st.columns(2)

with col_scan1:
    st.subheader("1️⃣ Checksheet VIN")
    sheet_file = st.camera_input("Capture Checksheet", key="cam_sheet")
    if sheet_file:
        img = Image.open(sheet_file)
        with st.spinner("Processing..."):
            res = extract_strict_pl1_vin(img)
            if res:
                st.session_state.checksheet_vin = res

with col_scan2:
    st.subheader("2️⃣ Car VIN")
    car_file = st.camera_input("Capture Car VIN", key="cam_car")
    if car_file:
        img = Image.open(car_file)
        with st.spinner("Processing..."):
            res = extract_strict_pl1_vin(img)
            if res:
                st.session_state.car_vin = res

st.divider()

# Result Comparison View
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN Result:", 
        value=st.session_state.checksheet_vin
    ).upper()

with col_input2:
    st.session_state.car_vin = st.text_input(
        "Car VIN Result:", 
        value=st.session_state.car_vin
    ).upper()

# Validation Banner
if st.session_state.checksheet_vin and st.session_state.car_vin:
    chk = st.session_state.checksheet_vin.strip()
    car = st.session_state.car_vin.strip()

    if chk == car:
        st.balloons()
        st.success(f"✅ MATCH CONFIRMED!\n\n`{chk}`")
    else:
        st.error(f"❌ MISMATCH DETECTED!\n\nChecksheet: `{chk}`\nCar VIN: `{car}`")
