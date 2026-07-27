import cv2
import re
import numpy as np
import av
import easyocr
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# -------------------------------------------------------------------
# 1. VIN VALIDATION UTILITIES
# -------------------------------------------------------------------
# VINs never contain I, O, or Q. Total length must be 17 characters.
VIN_PATTERN = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')

# ISO 3779 VIN Checksum verification weights & transliteration table
VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
VIN_TRANSLITERATION = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
    'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}

def is_valid_vin(vin: str) -> bool:
    """Verifies format and mathematical checksum (9th digit) of a VIN."""
    vin = vin.upper().strip()
    if not VIN_PATTERN.match(vin):
        return False
    
    # Calculate checksum
    total = sum(VIN_TRANSLITERATION[char] * VIN_WEIGHTS[idx] for idx, char in enumerate(vin))
    remainder = total % 11
    expected_check = 'X' if remainder == 10 else str(remainder)
    
    return vin[8] == expected_check

# -------------------------------------------------------------------
# 2. MODEL INITIALIZATION (Cached for performance)
# -------------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    # Load EasyOCR for English character detection
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

# -------------------------------------------------------------------
# 3. WEBRTC FRAME PROCESSOR (AR OVERLAY ENGINE)
# -------------------------------------------------------------------
class VINScannerProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_detected_vin = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Performance optimization: Process OCR every 5th frame to avoid video lag
        if self.frame_count % 5 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Restrict allowlist strictly to non-ambiguous VIN characters
            results = reader.readtext(
                gray, 
                allowlist='ABCDEFGHJKLMNPRSTUVWXYZ0123456789',
                paragraph=False
            )
            
            for (bbox, text, prob) in results:
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if len(cleaned_text) >= 17:
                    possible_vin = cleaned_text[:17]
                    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                    
                    if is_valid_vin(possible_vin):
                        color = (0, 255, 0)  # Green for verified valid VIN
                        label = f"VALID VIN: {possible_vin}"
                        self.last_detected_vin = possible_vin
                    else:
                        color = (0, 165, 255)  # Orange for format match with invalid checksum
                        label = f"RAW VIN: {possible_vin}"

                    # Draw AR Bounding Box & Text Overlay onto live camera stream
                    cv2.polylines(img, [pts], True, color, 3)
                    cv2.putText(img, label, (pts[0][0][0], max(30, pts[0][0][1] - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Persistent HUD banner on top of the stream displaying the last read VIN
        if self.last_detected_vin:
            cv2.rectangle(img, (20, 20), (450, 70), (0, 0, 0), -1)
            cv2.putText(img, f"DETECTED: {self.last_detected_vin}", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------------------------------------------
# 4. STREAMLIT UI SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="Real-time VIN AR Scanner", layout="centered")

st.title("🚗 Real-time VIN Scanner")
st.markdown("Point your camera at a vehicle door jamb sticker, dashboard plate, or window etching.")

# STUN server configuration for cross-network WebRTC streaming
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="vin-ar-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_processor_factory=VINScannerProcessor,
    media_stream_constraints={
        "video": {"facingMode": "environment"},  # Defaults to rear camera on mobile phones
        "audio": False
    },
    async_processing=True,
)

st.info("Tip: Keep camera steady about 6–12 inches away from the VIN plate under clean lighting.")