import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
from judge_logic import MovementJudge

# Page Config
st.set_page_config(page_title="CrossFit AI Judge", layout="wide")

st.title("🏋️ CrossFit Qualifier AI Validator")
st.markdown("Upload a video to analyze movement standards automatically.")

# Sidebar
st.sidebar.header("Configuration")
movement_option = st.sidebar.selectbox("Select Movement", ["Air Squat", "Thruster"])
confidence_threshold = st.sidebar.slider("AI Confidence", 0.5, 1.0, 0.5)

# File Uploader
uploaded_file = st.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov"])

# Main Processing Logic
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    st.sidebar.success("Video Uploaded Successfully")
    
    # Prep MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Prep Output Video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize Judge Logic
    judge = MovementJudge(movement_option)
    
    # Streamlit Placeholders
    st_frame = st.empty()
    st_data = st.empty()
    
    # Process Video Button
    if st.sidebar.button("Start Analysis"):
        with mp_pose.Pose(min_detection_confidence=confidence_threshold, min_tracking_confidence=confidence_threshold) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Recolor to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make Detection
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Run Judging Logic
                    reps, feedback = judge.process_frame(landmarks)
                    
                    # Draw UI on Frame
                    # Rep Box
                    cv2.rectangle(image, (0,0), (250, 73), (245,117,16), -1)
                    cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(reps), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Feedback Box
                    cv2.putText(image, 'STATUS', (100,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, feedback, (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                except Exception as e:
                    pass
                
                # Update Streamlit Frame
                st_frame.image(image, channels="BGR", use_column_width=True)
                
        st.success(f"Analysis Complete. Total Reps: {judge.counter}")
        cap.release()