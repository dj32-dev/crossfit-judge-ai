import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
import pandas as pd
import os
import time
from judge_logic import MovementJudge

# --- PAGE CONFIG ---
st.set_page_config(page_title="CrossFit Qualifier Judge", layout="wide")

st.title("🏆 CrossFit Qualifier AI Validator")
st.markdown("""
**Protocol:**
1. Paste the YouTube link of the qualifier video.
2. Configure the Workout Standards (Movement, Depth, etc.).
3. The AI will download, analyze, and generate a scorecard.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Workout Configuration")
video_url = st.sidebar.text_input("YouTube URL")

st.sidebar.header("2. Movement Standards")
movement_type = st.sidebar.selectbox("Select Movement", ["Thruster", "Air Squat"])

st.sidebar.subheader("Judging Criteria")
depth_strictness = st.sidebar.slider("Squat Depth Angle (Standard < 90)", 70, 100, 85)
lockout_strictness = st.sidebar.slider("Extension/Lockout Angle (Standard > 165)", 150, 180, 165)

max_duration = st.sidebar.slider("Analysis Duration (Seconds)", 10, 120, 60, help="Limit processing time for faster results.")

# --- HELPER FUNCTIONS ---
def download_video(url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': 'input_video.mp4',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'input_video.mp4'

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

# --- MAIN LOGIC ---
if st.sidebar.button("Analyze Video"):
    if not video_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        # 1. DOWNLOAD
        with st.spinner("Downloading video from YouTube..."):
            try:
                if os.path.exists("input_video.mp4"):
                    os.remove("input_video.mp4")
                video_path = download_video(video_url)
                st.success("Download Complete!")
            except Exception as e:
                st.error(f"Error downloading video: {e}")
                st.stop()

        # 2. ANALYSIS SETUP
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit processing to max_duration to prevent timeouts
        limit_frames = int(max_duration * fps)
        
        judge = MovementJudge(movement_type, depth_strictness, lockout_strictness)
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        st_frame = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 3. ANALYSIS LOOP
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count > limit_frames:
                    break
                
                # Timestamp Calculation
                current_time = frame_count / fps
                
                # Resize for speed (optional)
                frame = cv2.resize(frame, (640, 480))
                
                # Process
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    reps, noreps, feedback = judge.process_frame(landmarks, current_time)

                    # --- VISUAL OVERLAYS ---
                    # Status Box
                    color = (0, 255, 0) if feedback == "REP!" else (0, 0, 255) if feedback == "NO REP" else (245, 117, 16)
                    
                    cv2.rectangle(image, (0,0), (640, 60), color, -1)
                    
                    # Text Info
                    cv2.putText(image, f"REPS: {reps}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(image, f"NO-REPS: {noreps}", (200,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(image, feedback, (450,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                except:
                    pass

                # Update Streamlit Display
                # We skip frames for display performance (show every 3rd frame)
                if frame_count % 3 == 0:
                    st_frame.image(image, channels="BGR", use_column_width=True)
                
                # Update Progress
                progress = min(frame_count / limit_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing... Time: {format_time(current_time)}")
                
                frame_count += 1

        cap.release()
        
        # --- REPORT GENERATION ---
        st.success("Analysis Complete!")
        
        st.divider()
        st.subheader("📋 Validator Report")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Valid Reps", judge.reps)
        col2.metric("No-Reps Detected", judge.no_reps, delta_color="inverse")
        col3.metric("Video Time Analyzed", f"{max_duration}s")

        # Data Table
        if judge.event_log:
            df = pd.DataFrame(judge.event_log)
            # Format timestamp
            df['time_formatted'] = df['time'].apply(format_time)
            
            # Stylize the table
            def highlight_norep(val):
                return 'background-color: #ffcccc' if val == "NO REP" else ''

            st.dataframe(
                df[['time_formatted', 'type', 'reason']].style.map(lambda x: "color: red; font-weight: bold;" if x == "NO REP" else "", subset=['type']),
                use_container_width=True
            )
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report (CSV)", csv, "judge_report.csv", "text/csv")
        else:
            st.info("No reps or attempts detected in this timeframe.")
