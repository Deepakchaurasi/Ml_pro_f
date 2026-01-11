import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Air Sketch Solver")

# Use st.secrets or an environment variable for production!
API_KEY = "your api key" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite') # Gemini 1.5 is better for vision

# --- UI LAYOUT ---
st.title("🖐️ AI Low-Poly Air Math Solver")
col1, col2 = st.columns([3, 2])

with col1:
    run = st.checkbox('Run Webcam', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.header("AI Explanation")
    output_container = st.empty() # Container for dynamic text updates

# --- STATE MANAGEMENT ---
# This prevents the canvas from being wiped every frame
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'output_text' not in st.session_state:
    st.session_state.output_text = ""

# --- FUNCTIONS ---
def drawLowPolyHand(img, lmList):
    palm_triangles = [(0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17)]
    mesh_color, line_color = (180, 180, 180), (50, 50, 50)

    for p1, p2, p3 in palm_triangles:
        pts = np.array([lmList[p1][0:2], lmList[p2][0:2], lmList[p3][0:2]], np.int32)
        cv2.fillPoly(img, [pts], mesh_color)
        cv2.polylines(img, [pts], True, line_color, 2)

    fingers_indices = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for finger in fingers_indices:
        for i in range(len(finger) - 1):
            pt1, pt2 = lmList[finger[i]][0:2], lmList[finger[i+1]][0:2]
            cv2.line(img, pt1, pt2, mesh_color, 15)
            cv2.line(img, pt1, pt2, line_color, 3)
            cv2.circle(img, pt1, 8, line_color, cv2.FILLED)
        cv2.circle(img, lmList[finger[-1]][0:2], 8, line_color, cv2.FILLED)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7)
prev_pos = None

while run:
    success, img = cap.read()
    if not success:
        st.error("Webcam not detected.")
        break
        
    img = cv2.flip(img, 1)
    if st.session_state.canvas is None:
        st.session_state.canvas = np.zeros_like(img)

    black_board = np.full_like(img, 50) 
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        
        drawLowPolyHand(black_board, lmList)

        # 1. WRITE MODE: Index finger only
        if fingers == [0, 1, 0, 0, 0]:
            curr = lmList[8][0:2]
            if prev_pos is None: prev_pos = curr
            cv2.line(st.session_state.canvas, prev_pos, curr, (255, 0, 255), 10)
            prev_pos = curr
            cv2.putText(black_board, "Writing...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # 2. PAUSE: Index and Thumb
        elif fingers == [1, 1, 0, 0, 0]:
            prev_pos = None
            cv2.putText(black_board, "Paused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 3. CLEAR: Fist
        elif fingers == [0, 0, 0, 0, 0]:
            st.session_state.canvas = np.zeros_like(img)
            prev_pos = None
            cv2.putText(black_board, "Cleared!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 4. SOLVE: Open Hand
        elif fingers == [1, 1, 1, 1, 1]:
            cv2.putText(black_board, "Solving...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Send to AI (only if canvas isn't empty)
            if np.any(st.session_state.canvas):
                pil_img = Image.fromarray(st.session_state.canvas)
                try:
                    response = model.generate_content(["Solve this math problem exactly and show work.", pil_img])
                    st.session_state.output_text = response.text
                except Exception as e:
                    st.session_state.output_text = f"Error: {e}"
        else:
            prev_pos = None

    # Combine and Display
    combined = cv2.addWeighted(black_board, 1, st.session_state.canvas, 1, 0)
    FRAME_WINDOW.image(combined, channels="BGR")
    output_container.markdown(st.session_state.output_text)

cap.release()