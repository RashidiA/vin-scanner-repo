import streamlit as st
import easyocr
import re
from PIL import Image
import numpy as np

st.set_page_config(page_title="VIN Matching & Verification", layout="centered")

# Hide Streamlit throttling banner if present
st.markdown("""
    <style>
    div[data-testid="stNotification"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Cache EasyOCR model so it only loads once in memory
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def extract_vin(image):
    """Processes an image and extracts a 17-character VIN."""
    img_np = np.array(image)
    results = reader.readtext(img_np)
    
    # Clean text to keep only standard VIN characters (A-Z, 0-9, excluding I, O, Q)
    all_text = " ".join([res[1] for res in results]).upper()
    cleaned_text = re.sub(r'[^A-HJ-NPR-Z0-9]', '', all_text)
    
    # Search for a 17-character string
    vins = re.findall(r'[A-HJ-NPR-Z0-9]{17}', cleaned_text)
    return vins[0] if vins else None

st.title("🚗 VIN Comparison Tool")
st.write("Scan the printed checksheet first, then scan the car VIN to verify if they match.")

# Session state initialization
if 'checksheet_vin' not in st.session_state:
    st.session_state.checksheet_vin = None
if 'car_vin' not in st.session_state:
    st.session_state.car_vin = None

tab1, tab2 = st.tabs(["1️⃣ Scan Checksheet", "2️⃣ Scan Car VIN"])

# ----------------- TAB 1: Checksheet -----------------
with tab1:
    st.subheader("Step 1: Checksheet VIN")
    checksheet_img = st.camera_input("Take a photo of the printed checksheet VIN", key="cam_checksheet")
    
    if checksheet_img:
        image = Image.open(checksheet_img)
        detected_vin = extract_vin(image)
        
        if detected_vin:
            st.session_state.checksheet_vin = detected_vin
            st.success(f"Detected Checksheet VIN: `{detected_vin}`")
        else:
            st.warning("Could not clearly detect a 17-character VIN. You can manually enter or adjust below.")
            
    st.session_state.checksheet_vin = st.text_input(
        "Checksheet VIN (Manual Override):", 
        value=st.session_state.checksheet_vin or ""
    ).upper()

# ----------------- TAB 2: Car VIN -----------------
with tab2:
    st.subheader("Step 2: Car VIN")
    car_img = st.camera_input("Take a photo of the Car VIN (Door Jamb / Dashboard)", key="cam_car")
    
    if car_img:
        image = Image.open(car_img)
        detected_car_vin = extract_vin(image)
        
        if detected_car_vin:
            st.session_state.car_vin = detected_car_vin
            st.info(f"Detected Car VIN: `{detected_car_vin}`")
        else:
            st.warning("Could not clearly detect a 17-character VIN on the car.")

    st.session_state.car_vin = st.text_input(
        "Car VIN (Manual Override):", 
        value=st.session_state.car_vin or ""
    ).upper()

# ----------------- Comparison Logic -----------------
st.divider()
st.subheader("VIN Verification Result")

chk_vin = st.session_state.checksheet_vin
car_vin = st.session_state.car_vin

if chk_vin and car_vin:
    if chk_vin == car_vin:
        st.balloons()
        st.success(f"✅ MATCH CONFIRMED!\n\nChecksheet: `{chk_vin}`\nCar VIN: `{car_vin}`")
    else:
        st.error(f"❌ MISMATCH DETECTED!\n\nChecksheet: `{chk_vin}`\nCar VIN: `{car_vin}`")
else:
    st.info("Awaiting both VINs to perform validation.")
