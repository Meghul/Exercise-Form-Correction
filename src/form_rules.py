# src/form_rules.py
from utils import calculate_angle, dist

# Landmark indices (MediaPipe)
LM = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "NOSE": 0
}

# --- BICEP CURL RULE ---
def rule_bicep_elbow_angle(landmarks_px, side="left", top_thresh=50, bottom_thresh=150):
    if side == "left":
        sh = landmarks_px[LM["LEFT_SHOULDER"]]
        el = landmarks_px[LM["LEFT_ELBOW"]]
        wr = landmarks_px[LM["LEFT_WRIST"]]
    else:
        sh = landmarks_px[LM["RIGHT_SHOULDER"]]
        el = landmarks_px[LM["RIGHT_ELBOW"]]
        wr = landmarks_px[LM["RIGHT_WRIST"]]

    angle = calculate_angle(sh, el, wr)
    msg = ""
    if angle <= top_thresh:
        ok = True
        msg = f"{side} elbow: good top ({int(angle)}°)"
    elif angle >= bottom_thresh:
        ok = True
        msg = f"{side} elbow: good bottom ({int(angle)}°)"
    else:
        ok = False
        msg = f"{side} elbow: incomplete curl ({int(angle)}°)"
    return ok, msg, angle

# --- WRIST-SHOULDER ALIGNMENT ---
def rule_wrist_shoulder_alignment(landmarks_px, side="left", pixel_tol=25):
    if side == "left":
        sh = landmarks_px[LM["LEFT_SHOULDER"]]
        wr = landmarks_px[LM["LEFT_WRIST"]]
    else:
        sh = landmarks_px[LM["RIGHT_SHOULDER"]]
        wr = landmarks_px[LM["RIGHT_WRIST"]]

    dy = abs(sh[1] - wr[1])
    ok = dy <= pixel_tol
    msg = f"{side} wrist-shldr dy={dy}px"
    return ok, ("Aligned" if ok else "Not aligned"), dy

# --- BACK SYMMETRY ---
def rule_back_symmetry(landmarks_px, tol_pixels=30):
    l_sh = landmarks_px[LM["LEFT_SHOULDER"]]
    r_sh = landmarks_px[LM["RIGHT_SHOULDER"]]
    dy = abs(l_sh[1] - r_sh[1])
    ok = dy <= tol_pixels
    msg = f"Shoulder symmetry dy={dy}px"
    return ok, ("Symmetric" if ok else "Tilted"), dy

# --- TRICEP CURL RULE ---
def rule_tricep_extension(landmarks_px, side="left"):
    if side == "left":
        shoulder = landmarks_px[LM["LEFT_SHOULDER"]]
        elbow = landmarks_px[LM["LEFT_ELBOW"]]
        wrist = landmarks_px[LM["LEFT_WRIST"]]
    else:
        shoulder = landmarks_px[LM["RIGHT_SHOULDER"]]
        elbow = landmarks_px[LM["RIGHT_ELBOW"]]
        wrist = landmarks_px[LM["RIGHT_WRIST"]]

    angle = calculate_angle(shoulder, elbow, wrist)
    if angle > 160:
        return True, f"Good extension ({int(angle)}°)", angle
    else:
        return False, f"Extend your arm more ({int(angle)}°)", angle
