# Exercise Form Detection

This project is a **real-time exercise form correction tool** using computer vision. It analyzes webcam or video input to provide feedback on exercise posture, count repetitions, and highlight incorrect movements. The app supports exercises like **Bicep Curl** and **Tricep Extension**, with a tutorial video and live feedback.

---

## Features

- Real-time webcam feed with pose detection.
- Feedback on posture correctness using predefined rules:
  - **Bicep Curl:** Elbow angle, shoulder position.
  - **Tricep Extension:** Arm extension angle.
  - **Wrist-Shoulder Alignment** and **Back Symmetry** for general posture.
- Rep counter with reset option.
- Tutorial video playback for guidance.
- Visual overlay of landmarks and angles.
- Supports video input or live webcam feed.

---

## Posture Rules

- **Elbow Angle (Bicep Curl):**  
  Checks if the elbow reaches proper top and bottom angles during the curl movement. Gives feedback if the curl is incomplete.

- **Tricep Extension:**  
  Ensures full arm extension by calculating shoulder-elbow-wrist angle.

- **Wrist-Shoulder Alignment:**  
  Ensures wrists are aligned with shoulders vertically, preventing improper bending.

- **Back Symmetry:**  
  Detects if left and right shoulders are level to maintain a stable posture.

- **Other Rules:**  
  Additional rules like “No Shoulder Shrug” can be added for better accuracy.

---

## Logic Behind Rules

- All rules are **geometric and angle-based** using landmark coordinates extracted via pose detection.
- Angles are computed using 3 points: shoulder, elbow, and wrist.
- Thresholds are set to define good posture (e.g., elbow angle < 50° for top of curl, > 150° for bottom).
- Hysteresis is used in **rep counting** to avoid double counting.

---

## Challenges

- **Multiple Persons in Frame:** Currently, the system tracks only the primary person. Handling multiple people would require **individual pose tracking** and mapping each person to separate feedback.
- **Partial Visibility/Occlusions:** Landmark detection may fail when arms are out of frame.
- **Video Handling:** Switching tutorial videos can cause video decoding issues.

---

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd EXERCISE-FORM-DETECTION
### Create a virtual environment:
```bash
python -m venv venv
## Activate the virtual environment:

Windows:

.\venv\Scripts\activate


Linux / Mac:

source venv/bin/activate

Install required packages:
pip install -r requirements.txt

Run the Application
GUI:
python src/exercise_gui.py

Live webcam testing:
python src/live_exercise.py

Dependencies

Python 3.12+

OpenCV (opencv-python, opencv-contrib-python)

Mediapipe

NumPy

SciPy

Pillow

Tkinter (for GUI, usually included in Python)

Usage

Start the GUI.

Select an exercise from the dropdown menu.

Start the camera to receive live posture feedback.

Reset reps with the “Reset Reps” button.

Watch the tutorial video for guidance.

Stop the camera when finished.

Future Improvements

Multi-person detection and feedback.

Support for more exercises like squats, push-ups, lunges, lateral raises, and toe touches.

Advanced ML-based posture correction.

Save session data for tracking progress
```bash
python -m venv venv


## OUTPUT:

<img width="1920" height="1080" alt="Screenshot 2025-12-10 223016" src="https://github.com/user-attachments/assets/5e7443d4-7131-4c0a-a244-2a7a593b91bf" />
<img width="1920" height="1080" alt="Screenshot 2025-12-10 223047" src="https://github.com/user-attachments/assets/b77a96b2-7ff6-45e3-947a-e1dcef7632e6" />
