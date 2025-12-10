# src/exercise_gui.py
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from PIL import Image, ImageTk
from form_rules import rule_bicep_elbow_angle, rule_tricep_extension

# Map exercise to tutorial video path
EXERCISE_VIDEOS = {
    "bicep_curl": "D:/exercise-form-detection/videos/bicep_curl.mp4",
    "tricep_curl": "D:/exercise-form-detection/videos/tricep curl.mp4"
}

class ExerciseApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Exercise Form Corrector")
        self.master.geometry("1200x600")
        self.running = False
        self.cap = None
        self.video_cap = None
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        # --- Left frame: Camera feed ---
        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.left_frame, width=640, height=480)
        self.canvas.pack()

        # --- Middle frame: Steps & Feedback ---
        self.middle_frame = tk.Frame(master)
        self.middle_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        tk.Label(self.middle_frame, text="Select Exercise:").pack(pady=5)
        self.exercise_var = tk.StringVar(value="bicep_curl")
        self.dropdown = ttk.Combobox(self.middle_frame, textvariable=self.exercise_var)
        self.dropdown['values'] = ["bicep_curl", "tricep_curl"]
        self.dropdown.pack(pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.change_exercise)

        self.start_btn = tk.Button(self.middle_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(self.middle_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_btn.pack(pady=5)

        self.reset_btn = tk.Button(self.middle_frame, text="Reset Reps", command=self.reset_reps)
        self.reset_btn.pack(pady=5)

        # Feedback
        self.feedback_var = tk.StringVar(value="Feedback will appear here")
        self.feedback_label = tk.Label(self.middle_frame, textvariable=self.feedback_var, fg="blue")
        self.feedback_label.pack(pady=10)

        # Rep counter
        self.reps_var = tk.StringVar(value="Reps: 0")
        self.reps_label = tk.Label(self.middle_frame, textvariable=self.reps_var, fg="green", font=("Arial", 14))
        self.reps_label.pack(pady=10)

        # Exercise steps
        self.steps_text = tk.Text(self.middle_frame, height=20, width=30)
        self.steps_text.pack(pady=10)
        self.update_exercise_steps()

        # --- Right frame: Video playback ---
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(self.right_frame, text="Tutorial Video:").pack(pady=5)
        self.video_canvas = tk.Canvas(self.right_frame, width=320, height=240, bg="black")
        self.video_canvas.pack()

        # Internal variables
        self.current_angle = 0
        self.direction = 0  # 0=down, 1=up
        self.reps = 0

        self.current_video_path = EXERCISE_VIDEOS[self.exercise_var.get()]
        Thread(target=self.video_loop, daemon=True).start()

    def update_exercise_steps(self):
        ex = self.exercise_var.get()
        self.steps_text.delete("1.0", tk.END)
        if ex == "bicep_curl":
            steps = "Bicep Curl Steps:\n1. Stand straight\n2. Hold dumbbell\n3. Curl up slowly\n4. Lower slowly\n5. Keep elbows close to body"
        elif ex == "tricep_curl":
            steps = "Tricep Extension Steps:\n1. Hold dumbbell overhead\n2. Lower behind head\n3. Extend back up\n4. Keep elbows fixed"
        else:
            steps = "No steps available"
        self.steps_text.insert(tk.END, steps)

    def change_exercise(self, event):
        ex = self.exercise_var.get()
        self.update_exercise_steps()
        if ex in EXERCISE_VIDEOS:
            self.current_video_path = EXERCISE_VIDEOS[ex]

    def start_camera(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            Thread(target=self.camera_loop, daemon=True).start()
        else:
            messagebox.showinfo("Info", "Camera already running")

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.feedback_var.set("Camera stopped")
        self.reps_var.set(f"Reps: {self.reps}")

    def reset_reps(self):
        self.reps = 0
        self.direction = 0
        self.reps_var.set("Reps: 0")

    def camera_loop(self):
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            feedback_text = ""
            if results.pose_landmarks:
                landmarks_px = [(int(p.x * w), int(p.y * h)) for p in results.pose_landmarks.landmark]
                ex = self.exercise_var.get()
                if ex == "bicep_curl":
                    ok, msg, angle = rule_bicep_elbow_angle(landmarks_px, side="left")
                elif ex == "tricep_curl":
                    ok, msg, angle = rule_tricep_extension(landmarks_px, side="left")
                else:
                    ok, msg, angle = False, "Exercise not implemented", 0

                feedback_text = msg
                self.current_angle = angle

                if ex in ["bicep_curl", "tricep_curl"]:
                    if angle < 40 and self.direction == 0:
                        self.direction = 1
                    if angle > 160 and self.direction == 1:
                        self.direction = 0
                        self.reps += 1
                        self.reps_var.set(f"Reps: {self.reps}")

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"{int(angle)}°", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0,255,0) if ok else (0,0,255), 3)

            self.feedback_var.set(feedback_text)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0,0,anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def video_loop(self):
        last_video_path = None
        while True:
            if self.current_video_path != last_video_path:
                if self.video_cap:
                    self.video_cap.release()
                self.video_cap = cv2.VideoCapture(self.current_video_path)
                last_video_path = self.current_video_path

            if self.video_cap and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if not ret:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.resize(frame, (320, 240))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.video_canvas.image = imgtk

            cv2.waitKey(30)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseApp(root)
    root.mainloop()
