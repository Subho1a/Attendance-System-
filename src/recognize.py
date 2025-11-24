# recognize placeholder
# src/recognize.py
import os
import cv2
import pickle
import numpy as np
from src.detector import Detector
from src.embedder import Embedder
from src.utils import SVM_PATH, ENC_PATH, append_attendance, recently_marked, crop_from_box

def recognize(cam_index=0, threshold=0.75):
    if not os.path.exists(SVM_PATH) or not os.path.exists(ENC_PATH):
        raise RuntimeError('Model not trained. Run train first.')
    clf = pickle.load(open(SVM_PATH, 'rb'))
    le  = pickle.load(open(ENC_PATH, 'rb'))
    embedder = Embedder()

    with Detector() as det:
        cap = cv2.VideoCapture(cam_index)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, probs, landmarks = det.detect(frame)
            if boxes is not None:
                for box in boxes[:3]:
                    crop = crop_from_box(frame, box, size=embedder.model.input_size if hasattr(embedder.model, 'input_size') else 160)
                    if crop is None:
                        continue
                    emb = embedder.get_embedding(crop)
                    probs_ = clf.predict_proba(emb.reshape(1,-1))[0]
                    idx = int(np.argmax(probs_))
                    name = le.inverse_transform([idx])[0]
                    conf = float(probs_[idx])
                    x1,y1,x2,y2 = [int(round(v)) for v in box]
                    color = (0,255,0) if conf >= threshold else (0,0,255)
                    label = f"{name} {conf:.2f}" if conf >= threshold else f"Unknown {conf:.2f}"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                    if conf >= threshold and not recently_marked(name):
                        append_attendance(name, conf)
                        print('Marked:', name, conf)
            cv2.imshow('Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release(); cv2.destroyAllWindows()