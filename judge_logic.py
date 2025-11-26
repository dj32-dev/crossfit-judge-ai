import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

class MovementJudge:
    def __init__(self, movement_type, depth_threshold=85, extension_threshold=165):
        self.movement_type = movement_type
        self.depth_threshold = depth_threshold
        self.extension_threshold = extension_threshold
        
        self.reps = 0
        self.no_reps = 0
        self.stage = "start"
        self.feedback = "Setup"
        
        # Log structure: {'time': 1.23, 'type': 'Rep', 'reason': 'Valid'}
        self.event_log = [] 

    def process_frame(self, landmarks, timestamp):
        # Extract Coordinates
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate Angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # --- LOGIC ---
        
        # 1. THRUSTER / SQUAT
        if self.movement_type in ["Thruster", "Air Squat"]:
            
            # CHECK DEPTH
            if knee_angle < self.depth_threshold:
                self.stage = "bottom"
                self.feedback = "Depth Good"
            
            # CHECK EXTENSION (Rep completion)
            if self.stage == "bottom":
                # Check if standing up
                if knee_angle > self.extension_threshold and hip_angle > self.extension_threshold:
                    
                    is_valid = True
                    fail_reason = ""

                    # Thruster specific: Arms must be overhead
                    if self.movement_type == "Thruster":
                        if elbow_angle < self.extension_threshold:
                            is_valid = False
                            fail_reason = "Soft Elbows (No Lockout)"

                    if is_valid:
                        self.reps += 1
                        self.stage = "top"
                        self.feedback = "REP!"
                        self.event_log.append({'time': timestamp, 'type': 'Valid Rep', 'reason': '-'})
                    else:
                        # Only log no-rep if we haven't logged it for this specific rep yet
                        if self.feedback != "NO REP":
                            self.no_reps += 1
                            self.stage = "top" # Reset so we don't count it twice
                            self.feedback = "NO REP"
                            self.event_log.append({'time': timestamp, 'type': 'NO REP', 'reason': fail_reason})

        return self.reps, self.no_reps, self.feedback
