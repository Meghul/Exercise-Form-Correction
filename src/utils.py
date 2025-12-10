# src/utils.py
import math
import numpy as np
from scipy.signal import savgol_filter

def calculate_angle(a, b, c):
    """
    Calculate angle at point b formed by points a-b-c.
    Returns angle in degrees (0..180)
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang

def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def smooth_series(vals, method="savgol", window=11, poly=2):
    """
    Smooth 1D numpy array, fill NaNs first
    """
    v = np.array(vals, dtype=float)
    if np.isnan(v).any():
        idx = np.where(~np.isnan(v))[0]
        if len(idx) == 0:
            return v
        v[:idx[0]] = v[idx[0]]
        v[idx[-1]+1:] = v[idx[-1]]
        n_idx = np.isnan(v)
        if n_idx.any():
            v[n_idx] = np.interp(np.where(n_idx)[0], np.where(~n_idx)[0], v[~n_idx])
    if method == "savgol":
        wl = min(window, len(v) if len(v)%2==1 else len(v)-1)
        if wl < 5:
            return v
        try:
            return savgol_filter(v, wl, poly)
        except:
            return v
    elif method == "ema":
        alpha = 0.2
        out = np.zeros_like(v)
        out[0] = v[0]
        for i in range(1, len(v)):
            out[i] = alpha*v[i] + (1-alpha)*out[i-1]
        return out
    return v

def detect_reps_from_angle_series(angle_series, up_thresh=60, down_thresh=150, min_gap=10):
    """
    Hysteresis-based rep counter:
    up_thresh -> top, down_thresh -> bottom
    """
    state = "unknown"
    last_event_frame = -999
    rep_count = 0
    events = []
    for i, a in enumerate(angle_series):
        if a <= up_thresh:
            if state == "down" and (i - last_event_frame) > min_gap:
                state = "up"
                last_event_frame = i
            else:
                state = "up"
        elif a >= down_thresh:
            if state == "up" and (i - last_event_frame) > min_gap:
                rep_count += 1
                events.append(i)
                last_event_frame = i
                state = "down"
            else:
                state = "down"
        else:
            state = "moving"
    return rep_count, events
