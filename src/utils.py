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

os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

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

def append_attendance(name, conf):
    if not os.path.exists(ATT_CSV):
        with open(ATT_CSV, 'w') as f:
            f.write('timestamp,name,confidence\n')
    now = datetime.now().isoformat(sep=' ', timespec='seconds')
    with open(ATT_CSV, 'a') as f:
        f.write(f"{now},{name},{conf:.4f}\n")

def recently_marked(name, minutes=10):
    if not os.path.exists(ATT_CSV):
        return False
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with open(ATT_CSV, 'r') as f:
        lines = f.read().strip().splitlines()[1:]
    for ln in reversed(lines[-500:]):
        try:
            ts, nm, _ = ln.split(',', 2)
            t = datetime.fromisoformat(ts)
            if nm == name and t > cutoff:
                return True
        except Exception:
            continue
    return False