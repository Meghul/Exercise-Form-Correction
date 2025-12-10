# src/live_exercise.py
import cv2
import os
import numpy as np
from pose_detector import PoseDetector
from form_rules import rule_bicep_elbow_angle, rule_wrist_shoulder_alignment, rule_back_symmetry

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Pose detector
detector = PoseDetector()

# Rep counting state
left_angle_series = []
right_angle_series = []
fps = 30

# Rep counters
reps_left = 0
reps_right = 0

# Direction flags
dir_left = 0
dir_right = 0

def reset_reps():
    global reps_left, reps_right, dir_left, dir_right
    reps_left = 0
    reps_right = 0
    dir_left = 0
    dir_right = 0
    print("Reps have been reset!")

print("Press 'r' to reset reps, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = detector.detect(frame)

    feedback_msgs = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm]

        # Bicep curl angles
        ok_l, msg_l, angle_l = rule_bicep_elbow_angle(landmarks_px, side="left")
        ok_r, msg_r, angle_r = rule_bicep_elbow_angle(landmarks_px, side="right")
        feedback_msgs.append(msg_l)
        feedback_msgs.append(msg_r)

        # LEFT rep logic
        if angle_l > 160:
            dir_left = 0
        if angle_l < 40 and dir_left == 0:
            reps_left += 1
            dir_left = 1
            feedback_msgs.append("Left arm: GOOD REP!")

        # RIGHT rep logic
        if angle_r > 160:
            dir_right = 0
        if angle_r < 40 and dir_right == 0:
            reps_right += 1
            dir_right = 1
            feedback_msgs.append("Right arm: GOOD REP!")

        # Other rules
        ok_ws_l, ws_msg_l, _ = rule_wrist_shoulder_alignment(landmarks_px, side="left")
        ok_bs, bs_msg, _ = rule_back_symmetry(landmarks_px)
        feedback_msgs.append(ws_msg_l)
        feedback_msgs.append(bs_msg)

        # Draw pose
        cv2.putText(frame, f"L:{int(angle_l)}° R:{int(angle_r)}°", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps L:{reps_left} R:{reps_right}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        for i, msg in enumerate(feedback_msgs):
            cv2.putText(frame, msg, (20, 120 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        import mediapipe as mp
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    else:
        cv2.putText(frame, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Live Exercise Form Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty("Live Exercise Form Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('r'):
        reset_reps()  # Reset reps on pressing 'r'

cap.release()
cv2.destroyAllWindows()
detector.close()
