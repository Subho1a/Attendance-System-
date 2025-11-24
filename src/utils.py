# utils placeholder
# src/utils.py
import os
import cv2
import numpy as np
from datetime import datetime, timedelta

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE, 'data')
USERS_DIR = os.path.join(DATA_DIR, 'users')
MODELS_DIR = os.path.join(BASE, 'models')
ATT_DIR = os.path.join(BASE, 'attendance')
SVM_PATH = os.path.join(MODELS_DIR, 'svm_classifier.pkl')
ENC_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
ATT_CSV  = os.path.join(ATT_DIR, 'attendance.csv')

# Unified header for attendance CSV
ATT_HEADER = 'timestamp,student_id,name,confidence'

os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

# Ensure the attendance CSV exists and has the correct header
def ensure_attendance_csv():
    os.makedirs(ATT_DIR, exist_ok=True)
    if not os.path.exists(ATT_CSV):
        with open(ATT_CSV, 'w', encoding='utf-8', newline='') as f:
            f.write(ATT_HEADER + '\n')
    else:
        # If an old/mismatched header exists, rewrite header while preserving rows
        try:
            with open(ATT_CSV, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if first_line != ATT_HEADER:
                with open(ATT_CSV, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                if lines:
                    lines[0] = ATT_HEADER
                else:
                    lines = [ATT_HEADER]
                with open(ATT_CSV, 'w', encoding='utf-8', newline='') as f:
                    f.write('\n'.join(lines) + '\n')
        except Exception:
            # Non-critical: ignore header rewrite errors
            pass

def crop_from_box(frame, box, size=160):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        crop = cv2.resize(crop, (size, size))
    except Exception:
        return None
    # convert to RGB for Embedder
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop

# Append an attendance row (student_id defaults to name if not provided)
def append_attendance(name, conf, student_id=None):
    ensure_attendance_csv()
    if student_id is None:
        student_id = name
    now = datetime.now().isoformat(sep=' ', timespec='seconds')
    with open(ATT_CSV, 'a', encoding='utf-8', newline='') as f:
        f.write(f"{now},{student_id},{name},{conf:.4f}\n")

# Check if a person was recently marked to prevent duplicates
def recently_marked(name, minutes=10):
    ensure_attendance_csv()
    cutoff = datetime.now() - timedelta(minutes=minutes)
    try:
        with open(ATT_CSV, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        if not lines or len(lines) <= 1:
            return False
        data_lines = lines[1:]
    except Exception:
        return False
    for ln in reversed(data_lines[-500:]):
        try:
            fields = ln.split(',')
            if len(fields) < 3:
                continue
            ts = fields[0]
            # Prefer 'name' field if present at index 2 (timestamp,student_id,name,confidence)
            nm = fields[2] if len(fields) >= 3 else fields[1]
            t = datetime.fromisoformat(ts)
            if nm == name and t > cutoff:
                return True
        except Exception:
            continue
    return False