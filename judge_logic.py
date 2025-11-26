import mediapipe as mp
from utils import calculate_angle

mp_pose = mp.solutions.pose

class MovementJudge:
    def __init__(self, movement_type):
        self.movement_type = movement_type
        self.counter = 0
        self.stage = None # 'up' or 'down'
        self.feedback = "Setup"
        self.no_rep_count = 0
        self.rep_log = [] # Stores timestamp of reps

    def process_frame(self, landmarks):
        """
        Input: MediaPipe landmarks
        Output: Updated stage, counter, feedback
        """
        # Get Coordinates (Left side usually visible)
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate Angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # LOGIC: THRUSTER / AIR SQUAT
        if self.movement_type in ["Thruster", "Air Squat"]:
            
            # 1. DEPTH CHECK (Bottom of movement)
            # Standard: Hip crease below knee (approx < 80-90 degrees depending on camera angle)
            if knee_angle < 85:
                self.stage = "down"
                self.feedback = "Good Depth"

            # 2. EXTENSION CHECK (Top of movement)
            if self.stage == "down":
                # Check for Squat Extension
                is_hips_open = hip_angle > 160
                is_knees_open = knee_angle > 160
                
                if is_hips_open and is_knees_open:
                    # Specific Thruster Logic (Needs Overhead Lockout)
                    if self.movement_type == "Thruster":
                        if elbow_angle > 160: # Arms locked out
                            self.counter += 1
                            self.stage = "up"
                            self.feedback = "REP!"
                        elif elbow_angle < 140:
                            self.feedback = "Push Head Through!"
                    
                    # Just Squat Logic
                    else:
                        self.counter += 1
                        self.stage = "up"
                        self.feedback = "REP!"

        return self.counter, self.feedback