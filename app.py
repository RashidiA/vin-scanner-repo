import re
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

st.set_page_config(page_title="PL1 VIN Scanner", layout="wide")

# Hide Streamlit UI bloat & expand camera views
st.markdown(
    """
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    div[data-testid="stCameraInput"] { width: 100% !important; }
    div[data-testid="stCameraInput"] video { width: 100% !important; border: 3px solid #00E676; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rapid_ocr():
    return RapidOCR()


engine = load_rapid_ocr()

if "checksheet_vin" not in st.session_state:
    st.session_state.checksheet_vin = ""
if "car_vin" not in st.session_state:
    st.session_state.car_vin = ""


def process_pl1_vin(pil_image):
    # 1. Convert to numpy array
    img_np = np.array(pil_image.convert("RGB"))

    # 2. UN-MIRROR/FLIP HORIZONTALLY (Fixes front-camera selfie mirror effect)
    img_flipped = cv2.flip(img_np, 1)

    h, w, _ = img_flipped.shape

    # 3. CROP TO BOTTOM 35% OF IMAGE (Ignores table text like KEYFOB/REFLASH)
    bottom_crop = img_flipped[int(h * 0.65) :, :]

    # 4. PREPROCESSING (Increase contrast & grayscale)
    gray = cv2.cvtColor(bottom_crop, cv2.COLOR_RGB2GRAY)

    # Scale 2x for high resolution
    resized = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive Threshold to isolate text on pink/grey background
    thresh = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
    )
    processed_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # 5. RUN OCR (First on thresholded image, fallback to raw bottom crop)
    results, _ = engine(processed_rgb)
    if not results:
        results, _ = engine(bottom_crop)
    if not results:
        # Fallback to scanning the whole un-cropped flipped image
        results, _ = engine(img_flipped)

    if not results:
        return None

    # Combine all detected strings
    raw_text = "".join([item[1] for item in results]).upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text)

    if not cleaned:
        return None

    # Search for PL1 or common misreads + 14 trailing alphanumeric chars
    pl1_match = re.search(r"(PL1|PLI|PLL|P11|P1I|RL1|FL1)[A-Z0-9]{14}", cleaned)

    if pl1_match:
        suffix = pl1_match.group(0)[3:]
    else:
        # If no PL1 found directly, search for any 14-17 char sequence
        long_matches = re.findall(r"[A-Z0-9]{14,17}", cleaned)
        if long_matches:
            target = long_matches[0]
            suffix = target[3:17] if len(target) >= 17 else target[:14]
        else:
            suffix = cleaned[:14]

    # Clean illegal VIN characters (I -> 1, O/Q -> 0)
    corrected_suffix = []
    for char in suffix:
        if char == "I":
            corrected_suffix.append("1")
        elif char in ["O", "Q"]:
            corrected_suffix.append("0")
        else:
            corrected_suffix.append(char)

    final_suffix = "".join(corrected_suffix)

    # FORCE PL1 PREFIX
    final_vin = f"PL1{final_suffix}"
    return final_vin[:17]


# Streamlit UI
st.title("🚗 PL1 VIN Matcher")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Checksheet VIN")
    sheet_file = st.camera_input("Hold VIN at the BOTTOM of the frame", key="cam_sheet")
    if sheet_file:
        img = Image.open(sheet_file)
        with st.spinner("Extracting VIN..."):
            res = process_pl1_vin(img)
            if res:
                st.session_state.checksheet_vin = res
                st.success(f"Detected: `{res}`")

with col2:
    st.subheader("2️⃣ Car VIN")
    car_file = st.camera_input("Take photo of Car VIN", key="cam_car")
    if car_file:
        img = Image.open(car_file)
        with st.spinner("Extracting VIN..."):
            res = process_pl1_vin(img)
            if res:
                st.session_state.car_vin = res
                st.success(f"Detected: `{res}`")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN:", value=st.session_state.checksheet_vin
    ).upper()
with c2:
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
