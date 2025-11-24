# src/gui_pyqt.py
"""
PyQt5 GUI — FaceNet + SVM (MTCNN via facenet-pytorch)

Behavior:
- No saving of face crop images (crops folder removed)
- Attendance rows appended to attendance/attendance.csv with header:
    timestamp,student_id,name,confidence
- CSV is auto-created with header if missing
- Prevent duplicate attendance within 1 minute
- Optional name -> student_id mapping via models/id_map.json

Run:
    venv\Scripts\activate
    pip install PyQt5
    python -m src.gui_pyqt
"""

import os
import sys
import time
import json
import traceback
import numpy as np
from datetime import datetime, timedelta

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import cv2

# project modules
from src.detector import Detector
from src.embedder import Embedder
from src.train import train_svm
from src.utils import (
    USERS_DIR,
    crop_from_box,
    SVM_PATH,
    ENC_PATH,
    ATT_CSV,
    ATT_DIR,
    recently_marked,
    ensure_attendance_csv,  # added import
)

# Optional mapping file: name -> student_id
ID_MAP_PATH = os.path.join("models", "id_map.json")


# ----------------- Helpers -----------------
def cv2_to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)


def ensure_attendance_csv_gui():
    """Ensure attendance CSV exists and has the desired header:
       timestamp,student_id,name,confidence
    """
    ensure_attendance_csv()


def remove_attendance_crops_folder():
    """If an old attendance/crops folder exists, remove it (user requested no crops)."""
    crops_dir = os.path.join(ATT_DIR, "crops")
    try:
        if os.path.exists(crops_dir) and os.path.isdir(crops_dir):
            # remove files then folder
            for fname in os.listdir(crops_dir):
                try:
                    os.remove(os.path.join(crops_dir, fname))
                except Exception:
                    pass
            try:
                os.rmdir(crops_dir)
            except Exception:
                pass
    except Exception:
        pass


# ----------------- Camera worker -----------------
class CameraWorker(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(np.ndarray)
    overlay_signal = QtCore.pyqtSignal(list)
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self._running = False
        self.detector = None
        self.embedder = None
        self.clf = None
        self.le = None
        self.recognition_enabled = False
        self.threshold = 0.75

    def start_camera(self):
        self._running = True
        if not self.isRunning():
            self.start()

    def stop_camera(self):
        self._running = False
        # allow thread to stop
        self.wait(1000)

    def load_models(self):
        try:
            if self.embedder is None:
                self.embedder = Embedder()
            if (self.clf is None) or (self.le is None):
                if os.path.exists(SVM_PATH) and os.path.exists(ENC_PATH):
                    import pickle
                    with open(SVM_PATH, "rb") as f:
                        self.clf = pickle.load(f)
                    with open(ENC_PATH, "rb") as f:
                        self.le = pickle.load(f)
                    self.log_signal.emit("SVM and label encoder loaded.")
                else:
                    self.log_signal.emit("SVM / encoder missing; recognition disabled until trained.")
        except Exception as e:
            self.log_signal.emit("Model load error: " + str(e))
            self.clf = None
            self.le = None

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.log_signal.emit("Camera failed to open")
            return

        # start Detector
        try:
            self.detector = Detector()
            self.detector.__enter__()
        except Exception as e:
            self.log_signal.emit("Detector init error: " + str(e))
            self.detector = None

        self.log_signal.emit("Camera started")
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("Camera read failed")
                break

            # emit raw frame for display
            try:
                self.frame_signal.emit(frame.copy())
            except Exception:
                pass

            annotations = []
            try:
                if self.detector is not None:
                    det_out = self.detector.detect(frame)
                    if det_out is None:
                        boxes, probs, landmarks = None, None, None
                    else:
                        boxes, probs, landmarks = det_out

                    if boxes is not None and len(boxes) > 0:
                        if self.recognition_enabled:
                            if self.embedder is None or self.clf is None or self.le is None:
                                self.load_models()

                            for box in boxes:
                                crop = crop_from_box(frame, box, size=160)  # crop is RGB
                                if crop is None:
                                    annotations.append((box, "", 0.0))
                                    continue
                                try:
                                    emb = self.embedder.get_embedding(crop)
                                except Exception as e:
                                    self.log_signal.emit("Embedding error: " + str(e))
                                    annotations.append((box, "", 0.0))
                                    continue

                                if self.clf is None or self.le is None:
                                    annotations.append((box, "Unknown", 0.0))
                                    continue

                                probs_ = self.clf.predict_proba(emb.reshape(1, -1))[0]
                                idx = int(np.argmax(probs_))
                                conf = float(probs_[idx])
                                name = self.le.inverse_transform([idx])[0]
                                label = f"{name} {conf:.2f}" if conf >= self.threshold else f"Unknown {conf:.2f}"
                                annotations.append((box, label, conf))
                        else:
                            for box in boxes:
                                annotations.append((box, "", 0.0))
                else:
                    annotations = []
            except Exception as e:
                self.log_signal.emit("CameraWorker detect/recognize error: " + str(e))
                annotations = []

            try:
                self.overlay_signal.emit(annotations)
            except Exception:
                pass

            self.msleep(20)

        # cleanup
        try:
            cap.release()
        except:
            pass
        try:
            if self.detector is not None:
                self.detector.__exit__(None, None, None)
        except:
            pass
        self.log_signal.emit("Camera stopped")


# ----------------- Embedding worker -----------------
class EmbeddingWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(dict)

    def __init__(self, user_dir):
        super().__init__()
        self.user_dir = user_dir

    def run(self):
        try:
            imgs_dir = os.path.join(self.user_dir, "imgs")
            if not os.path.exists(imgs_dir):
                self.log_signal.emit("No images folder: " + imgs_dir)
                self.finished_signal.emit({"ok": False})
                return

            paths = sorted([os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.lower().endswith(".jpg")])
            if len(paths) == 0:
                self.log_signal.emit("No images to embed")
                self.finished_signal.emit({"ok": False})
                return

            self.log_signal.emit(f"Embedding {len(paths)} images...")
            embedder = Embedder()
            embs = []
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    emb = embedder.get_embedding(rgb)
                except Exception as e:
                    self.log_signal.emit("Embed error for " + p + ": " + str(e))
                    continue
                embs.append(emb)

            if len(embs) == 0:
                self.log_signal.emit("No embeddings generated")
                self.finished_signal.emit({"ok": False})
                return

            embs = np.vstack(embs)
            np.save(os.path.join(self.user_dir, "embeddings.npy"), embs)
            self.log_signal.emit(f"Saved {len(embs)} embeddings for {os.path.basename(self.user_dir)}")
            self.finished_signal.emit({"ok": True})
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit("EmbeddingWorker exception: " + str(e) + "\n" + tb)
            self.finished_signal.emit({"ok": False})


# ----------------- Attendance worker (no crops saved) -----------------
class AttendanceWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(dict)
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, frame_bgr):
        super().__init__()
        self.frame = frame_bgr.copy()
        self.detector = None
        self.embedder = None
        self.clf = None
        self.le = None

    def load_classifier(self):
        if os.path.exists(SVM_PATH) and os.path.exists(ENC_PATH):
            import pickle
            try:
                with open(SVM_PATH, "rb") as f:
                    self.clf = pickle.load(f)
                with open(ENC_PATH, "rb") as f:
                    self.le = pickle.load(f)
                return True
            except Exception as e:
                self.log_signal.emit("Failed to load classifier: " + str(e))
                return False
        else:
            self.log_signal.emit("SVM or label encoder not found.")
            return False

    def run(self):
        try:
            # ensure csv exists and crops folder removed
            ensure_attendance_csv_gui()
            remove_attendance_crops_folder()

            # detection
            det = Detector()
            det.__enter__()
            det_out = det.detect(self.frame)
            if det_out is None:
                boxes, probs, landmarks = None, None, None
            else:
                boxes, probs, landmarks = det_out
            det.__exit__(None, None, None)

            if boxes is None or len(boxes) == 0:
                self.log_signal.emit("No face detected for attendance.")
                self.finished_signal.emit({"ok": False, "reason": "no_face"})
                return

            # pick largest
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            idx = int(np.argmax(areas))
            box = boxes[idx]

            # crop (crop_from_box returns RGB)
            crop = crop_from_box(self.frame, box, size=160)
            if crop is None:
                self.log_signal.emit("Crop failed")
                self.finished_signal.emit({"ok": False, "reason": "crop_failed"})
                return

            # embedding (Embedder expects RGB)
            self.embedder = Embedder()
            try:
                emb = self.embedder.get_embedding(crop)
            except Exception as e:
                self.log_signal.emit("Embedding failed: " + str(e))
                self.finished_signal.emit({"ok": False, "reason": "embed_failed", "error": str(e)})
                return

            # classifier
            ok = self.load_classifier()
            if not ok:
                self.log_signal.emit("Classifier unavailable.")
                self.finished_signal.emit({"ok": False, "reason": "no_classifier"})
                return

            probs = self.clf.predict_proba(emb.reshape(1, -1))[0]
            idx_cls = int(np.argmax(probs))
            conf = float(probs[idx_cls])
            name = self.le.inverse_transform([idx_cls])[0]

            # check recent duplicates
            if recently_marked(name, minutes=1):
                self.log_signal.emit(f"{name} recently marked; skipping duplicate.")
                self.finished_signal.emit({"ok": False, "reason": "recent_duplicate", "name": name})
                return

            # map to student id if map exists
            student_id = name
            try:
                if os.path.exists(ID_MAP_PATH):
                    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
                        id_map = json.load(f)
                    if isinstance(id_map, dict) and name in id_map:
                        student_id = id_map[name]
            except Exception as e:
                self.log_signal.emit("ID map load error: " + str(e))

            # append attendance row in desired format:
            # timestamp,student_id,name,confidence
            try:
                ts = datetime.now().isoformat(sep=" ", timespec="seconds")
                line = f"{ts},{student_id},{name},{conf:.4f}\n"
                with open(ATT_CSV, "a", encoding="utf-8", newline="") as f:
                    f.write(line)
            except Exception as e:
                self.log_signal.emit("Write attendance failed: " + str(e))
                self.finished_signal.emit({"ok": False, "reason": "write_failed", "error": str(e)})
                return

            self.log_signal.emit(f"Attendance marked: {name} ({student_id}) conf={conf:.3f}")
            self.finished_signal.emit({"ok": True, "name": name, "student_id": student_id, "confidence": conf})

        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit("Attendance worker exception: " + str(e) + "\n" + tb)
            self.finished_signal.emit({"ok": False, "reason": "exception", "error": str(e)})


# ----------------- MainWindow (GUI) -----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance — FaceNet + SVM (PyQt)")
        self.resize(1050, 650)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left: video preview
        left = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(720, 540)
        self.video_label.setStyleSheet("background:black;")
        left.addWidget(self.video_label)

        bottom = QtWidgets.QHBoxLayout()
        self.btn_stopcam = QtWidgets.QPushButton("Stop Camera")
        self.btn_stopcam.clicked.connect(self.on_stop_camera)
        bottom.addWidget(self.btn_stopcam)

        self.btn_snapshot = QtWidgets.QPushButton("Snapshot")
        self.btn_snapshot.clicked.connect(self.on_snapshot)
        bottom.addWidget(self.btn_snapshot)
        left.addLayout(bottom)

        # Right: controls
        right = QtWidgets.QVBoxLayout()

        # Register panel
        reg_box = QtWidgets.QGroupBox("Register")
        reg_form = QtWidgets.QFormLayout()
        self.input_name = QtWidgets.QLineEdit()
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(5, 500)
        self.spin_n.setValue(50)
        reg_form.addRow("Name:", self.input_name)
        reg_form.addRow("Images:", self.spin_n)
        self.btn_reg = QtWidgets.QPushButton("Register (Webcam)")
        self.btn_reg.clicked.connect(self.on_register)
        reg_form.addRow(self.btn_reg)
        self.btn_reg_file = QtWidgets.QPushButton("Register From File")
        self.btn_reg_file.clicked.connect(self.on_register_file)
        reg_form.addRow(self.btn_reg_file)
        reg_box.setLayout(reg_form)
        right.addWidget(reg_box)

        # Train panel
        train_box = QtWidgets.QGroupBox("Train SVM")
        tlay = QtWidgets.QVBoxLayout()
        self.btn_train = QtWidgets.QPushButton("Train")
        self.btn_train.clicked.connect(self.on_train)
        tlay.addWidget(self.btn_train)
        train_box.setLayout(tlay)
        right.addWidget(train_box)

        # Recognize / Attendance
        recog_box = QtWidgets.QGroupBox("Recognize / Attendance")
        rform = QtWidgets.QFormLayout()
        self.spin_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_thresh.setRange(0.1, 0.99)
        self.spin_thresh.setValue(0.75)
        rform.addRow("Threshold:", self.spin_thresh)
        self.btn_recog = QtWidgets.QPushButton("Start Recognition")
        self.btn_recog.clicked.connect(self.on_toggle_recognize)
        rform.addRow(self.btn_recog)

        self.btn_attendance = QtWidgets.QPushButton("Take Attendance (One-shot)")
        self.btn_attendance.clicked.connect(self.on_take_attendance)
        rform.addRow(self.btn_attendance)

        recog_box.setLayout(rform)
        right.addWidget(recog_box)

        # Logs
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log)

        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=1)

        # state
        self._annotations = []
        self.registering = False
        self.register_target = None
        self.register_target_n = 0
        self.register_count = 0
        self._last_capture_time = 0.0
        self.embedding_worker = None
        self.attendance_worker = None
        self.recognition_active = False

        # camera worker
        self.cam_worker = None
        self._start_camera_worker()

        # last frame
        self.last_frame = None

        # ensure CSV and remove crops (one-time)
        ensure_attendance_csv_gui()
        remove_attendance_crops_folder()

    # start camera worker
    def _start_camera_worker(self):
        self.cam_worker = CameraWorker()
        self.cam_worker.frame_signal.connect(self.update_frame)
        self.cam_worker.overlay_signal.connect(self.update_overlay)
        self.cam_worker.log_signal.connect(self.log_msg)
        self.cam_worker.start_camera()

    def log_msg(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")

    def update_frame(self, frame):
        # keep a copy (BGR)
        self.last_frame = frame.copy()

        canvas = frame.copy()
        for (box, label, conf) in getattr(self, "_annotations", []):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            color = (0,255,0) if ("Unknown" not in label and label != "") else (0,0,255)
            if self.registering:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # registration capture logic (unchanged)
        if self.registering and len(self._annotations) > 0:
            now = time.time()
            if now - self._last_capture_time > 0.12:
                box, _, _ = self._annotations[0]
                crop = crop_from_box(frame, box, size=160)
                if crop is not None:
                    user_dir = os.path.join(USERS_DIR, self.register_target)
                    imgs_dir = os.path.join(user_dir, "imgs")
                    os.makedirs(imgs_dir, exist_ok=True)
                    pth = os.path.join(imgs_dir, f"{self.register_count:03d}.jpg")
                    # save crop as BGR; crop_from_box returns RGB
                    cv2.imwrite(pth, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    self.register_count += 1
                    self._last_capture_time = now
                    self.log_msg(f"Captured {self.register_count}/{self.register_target_n}")
                    if self.register_count >= self.register_target_n:
                        self._finish_registration()

        qimg = cv2_to_qimage(canvas)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def update_overlay(self, annotations):
        self._annotations = annotations

    # snapshot
    def on_snapshot(self):
        if self.last_frame is not None:
            p = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(p, self.last_frame)
            self.log_msg("Snapshot saved: " + p)
        else:
            self.log_msg("No frame available")

    # camera toggle
    def on_stop_camera(self):
        if self.cam_worker and self.cam_worker.isRunning():
            self.cam_worker.stop_camera()
            self.btn_stopcam.setText("Start Camera")
            self.log_msg("Camera stopped")
        else:
            self._start_camera_worker()
            self.btn_stopcam.setText("Stop Camera")
            self.log_msg("Camera started")

    # registration
    def on_register(self):
        name = self.input_name.text().strip()
        n = int(self.spin_n.value())
        if not name:
            self.log_msg("Enter a name")
            return
        if self.registering:
            self.log_msg("Already registering")
            return
        if self.recognition_active:
            self.cam_worker.recognition_enabled = False
            self.recognition_active = False
            self.btn_recog.setText("Start Recognition")
        # Prepare user image directory fresh for this session to avoid embedding stale images
        user_dir = os.path.join(USERS_DIR, name)
        imgs_dir = os.path.join(user_dir, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)
        removed = 0
        for fn in os.listdir(imgs_dir):
            if fn.lower().endswith(".jpg"):
                try:
                    os.remove(os.path.join(imgs_dir, fn))
                    removed += 1
                except Exception:
                    pass
        # Remove stale embeddings if present
        try:
            emb_path = os.path.join(user_dir, "embeddings.npy")
            if os.path.exists(emb_path):
                os.remove(emb_path)
        except Exception:
            pass
        if removed:
            self.log_msg(f"Cleared {removed} previous images for {name}")
        self.btn_reg.setEnabled(False)
        self.btn_reg_file.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_recog.setEnabled(False)
        self.registering = True
        self.register_target = name
        self.register_target_n = n
        self.register_count = 0
        self._last_capture_time = 0.0
        self.log_msg(f"Started registration for {name}: capturing {n} images")

    def _finish_registration(self):
        self.registering = False
        self.log_msg("Finished capturing; computing embeddings...")
        user_dir = os.path.join(USERS_DIR, self.register_target)
        self.embedding_worker = EmbeddingWorker(user_dir)
        self.embedding_worker.log_signal.connect(self.log_msg)
        self.embedding_worker.finished_signal.connect(self._on_embedding_done)
        self.embedding_worker.start()

    def _on_embedding_done(self, res):
        self.log_msg(f"Embedding worker finished: {res}")
        self.btn_reg.setEnabled(True)
        self.btn_reg_file.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_recog.setEnabled(True)

    # register from file
    def on_register_file(self):
        name = self.input_name.text().strip()
        if not name:
            self.log_msg("Enter a name")
            return
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", os.getcwd(), "Images (*.png *.jpg *.jpeg)")
        if not fname:
            return
        user_dir = os.path.join(USERS_DIR, name)
        imgs_dir = os.path.join(user_dir, "imgs"); os.makedirs(imgs_dir, exist_ok=True)
        dst = os.path.join(imgs_dir, "000.jpg")
        import shutil
        shutil.copyfile(fname, dst)
        self.log_msg(f"Copied {fname} -> {dst}")
        try:
            embedder = Embedder()
            img = cv2.imread(dst)
            emb = embedder.get_embedding(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            np.save(os.path.join(user_dir, "embeddings.npy"), emb.reshape(1, -1))
            self.log_msg("Saved embedding for file registration")
        except Exception as e:
            self.log_msg("Register-from-file error: " + str(e))

    # train svm
    def on_train(self):
        try:
            self.log_msg("Training SVM...")
            train_svm()
            self.log_msg("Training completed")
        except Exception as e:
            self.log_msg("Training error: " + str(e))

    # recognition toggle
    def on_toggle_recognize(self):
        if not self.recognition_active:
            if not (self.cam_worker and self.cam_worker.isRunning()):
                self._start_camera_worker()
            self.cam_worker.load_models()
            self.cam_worker.threshold = float(self.spin_thresh.value())
            self.cam_worker.recognition_enabled = True
            self.recognition_active = True
            self.btn_recog.setText("Stop Recognition")
            self.log_msg("Recognition started")
        else:
            self.cam_worker.recognition_enabled = False
            self.recognition_active = False
            self.btn_recog.setText("Start Recognition")
            self.log_msg("Recognition stopped")

    # one-shot attendance
    def on_take_attendance(self):
        if self.last_frame is None:
            self.log_msg("No frame available to take attendance")
            return
        # Prevent starting multiple workers simultaneously
        if self.attendance_worker and self.attendance_worker.isRunning():
            self.log_msg("Attendance already in progress")
            return
        self.btn_attendance.setEnabled(False)
        self.log_msg("Taking attendance (one-shot)...")
        self.attendance_worker = AttendanceWorker(self.last_frame)
        self.attendance_worker.log_signal.connect(self.log_msg)

        def finished_handler(res):
            # Clear reference so GC won't destroy an active thread prematurely
            self.attendance_worker = None
            self.btn_attendance.setEnabled(True)
            if res.get("ok"):
                name = res.get("name", "")
                sid = res.get("student_id", "")
                conf = res.get("confidence", 0.0)
                self.log_msg(f"Attendance recorded: {name} ({sid}) conf={conf:.3f}")
                QMessageBox.information(self, "Attendance", f"Marked: {name}\nID: {sid}\nConfidence: {conf:.3f}")
            else:
                reason = res.get("reason", "unknown")
                if reason == "recent_duplicate":
                    self.log_msg("Duplicate prevented (recently marked).")
                    QMessageBox.warning(self, "Attendance", f"Duplicate: already marked recently.")
                elif reason == "no_face":
                    self.log_msg("No face detected for attendance.")
                    QMessageBox.warning(self, "Attendance", "No face detected.")
                else:
                    err = res.get("error", "")
                    self.log_msg(f"Attendance failed: {reason} {err}")
                    QMessageBox.critical(self, "Attendance", f"Failed: {reason}\n{err}")

        self.attendance_worker.finished_signal.connect(finished_handler)
        self.attendance_worker.start()

    def closeEvent(self, event):
        try:
            if self.cam_worker:
                self.cam_worker.stop_camera()
            if self.attendance_worker and self.attendance_worker.isRunning():
                # Wait briefly for attendance worker to finish to avoid QThread warnings
                self.attendance_worker.wait(1000)
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
