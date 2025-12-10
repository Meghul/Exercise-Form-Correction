# src/pose_detector.py
import mediapipe as mp
import cv2

class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame_bgr):
        """
        Input: BGR frame (numpy array).
        Output: Mediapipe results object (has .pose_landmarks)
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        return results

    def close(self):
        self.pose.close()
