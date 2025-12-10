# src/main.py
import cv2
import os
import numpy as np
import pandas as pd
from pose_detector import PoseDetector
from form_rules import (
    rule_bicep_elbow_angle,
    rule_wrist_shoulder_alignment,
    rule_back_symmetry,
    rule_no_shoulder_shrug
)
from utils import smooth_series, detect_reps_from_angle_series

# Optional MLflow logging toggle
USE_MLFLOW = False
if USE_MLFLOW:
    import mlflow

# --- CONFIG --- #
VIDEO_IN = "videos/bicep_curl.mp4"  # Change this to your video
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_OUT = os.path.join(OUTPUT_DIR, "annotated_bicep.mp4")
CSV_OUT = os.path.join(OUTPUT_DIR, "angles.csv")

# Map exercises to rules
EXERCISES = {
    "bicep_curl": [rule_bicep_elbow_angle, rule_back_symmetry, rule_no_shoulder_shrug],
    "lateral_raise": [rule_wrist_shoulder_alignment, rule_back_symmetry, rule_no_shoulder_shrug],
    "push_up": [rule_back_symmetry, rule_no_shoulder_shrug]  # example
}


def process_video(input_path, output_path, csv_path, exercise="bicep_curl", use_mlflow=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    detector = PoseDetector()
    records = []
    angle_history = {"left": [], "right": []}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)
        rec = {"frame": frame_idx}
        feedbacks = []

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm]

            # Apply rules
            for rule in EXERCISES[exercise]:
                # Rules that need 'side'
                if rule.__name__ in [
                    "rule_bicep_elbow_angle",
                    "rule_wrist_shoulder_alignment",
                    "rule_no_shoulder_shrug"
                ]:
                    for side in ["left", "right"]:
                        ok, msg, val = rule(landmarks_px, side=side)
                        feedbacks.append(msg)
                        if rule.__name__ == "rule_bicep_elbow_angle":
                            angle_history[side].append(val)
                # Rules that do NOT need 'side'
                else:
                    ok, msg, val = rule(landmarks_px)
                    feedbacks.append(msg)

            # Overlay feedback
            for i, fb in enumerate(feedbacks[:6]):  # limit number of lines
                cv2.putText(frame, fb, (20, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            rec.update({
                "left_elbow_angle": angle_history["left"][-1] if angle_history["left"] else np.nan,
                "right_elbow_angle": angle_history["right"][-1] if angle_history["right"] else np.nan
            })
        else:
            rec.update({"left_elbow_angle": np.nan, "right_elbow_angle": np.nan})
            cv2.putText(frame, "No pose detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        records.append(rec)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    detector.close()

    # Save data
    df = pd.DataFrame(records)

    # Smooth angles
    for col in ["left_elbow_angle", "right_elbow_angle"]:
        if col in df.columns:
            sm = smooth_series(df[col].to_numpy(), method="savgol", window=11, poly=2)
            df[col + "_smoothed"] = sm

    df.to_csv(csv_path, index=False)

    # Detect reps on smoothed angles
    left_series = df["left_elbow_angle_smoothed"].fillna(method="bfill").fillna(method="ffill").to_numpy()
    right_series = df["right_elbow_angle_smoothed"].fillna(method="bfill").fillna(method="ffill").to_numpy()

    reps_left, events_left = detect_reps_from_angle_series(left_series, up_thresh=60, down_thresh=150, min_gap=int(fps*0.3))
    reps_right, events_right = detect_reps_from_angle_series(right_series, up_thresh=60, down_thresh=150, min_gap=int(fps*0.3))

    summary = {
        "frames": len(df),
        "reps_left": reps_left,
        "reps_right": reps_right,
        "left_mean_angle": float(np.nanmean(left_series)),
        "right_mean_angle": float(np.nanmean(right_series))
    }

    # Optional MLflow logging
    if use_mlflow:
        mlflow.set_experiment("form_eval")
        with mlflow.start_run():
            mlflow.log_params({"input_video": input_path})
            for k, v in summary.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(output_path, artifact_path="annotated_videos")
            mlflow.log_artifact(csv_path, artifact_path="csvs")

    return df, summary


if __name__ == "__main__":
    df, summary = process_video(VIDEO_IN, VIDEO_OUT, CSV_OUT, exercise="bicep_curl", use_mlflow=USE_MLFLOW)
    print("Summary:", summary)
    print("Saved:", VIDEO_OUT, CSV_OUT)
