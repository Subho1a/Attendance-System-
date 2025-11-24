# detector placeholder
# src/detector.py
# MTCNN wrapper using facenet-pytorch for detection/alignment

from facenet_pytorch import MTCNN
import torch
import numpy as np

class Detector:
    def __init__(self, device=None, image_size=160):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.mtcnn = None

    def __enter__(self):
        # keep_all True so detect() returns arrays for multiple faces
        self.mtcnn = MTCNN(keep_all=True, device=self.device, image_size=self.image_size)
        return self

    def __exit__(self, exc_type, exc, tb):
        # facenet-pytorch MTCNN has no explicit close - delete reference
        if self.mtcnn is not None:
            del self.mtcnn
            self.mtcnn = None

    def detect(self, img):
        """Detect faces in an image.
        Args:
            img: PIL image or numpy array (H,W,3) OpenCV BGR or RGB works with mtcnn.detect
        Returns:
            boxes: Nx4 numpy array of (x1,y1,x2,y2) in pixel coords or None
            probs: Nx1 array of detection probabilities or None
            landmarks: Nx5x2 array of landmark coords or None
        """
        if self.mtcnn is None:
            raise RuntimeError('Detector not started. Use with Detector() as d:')
        # MTCNN.detect accepts numpy arrays (RGB or BGR); it returns boxes, probs, points
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        # boxes can be None
        return boxes, probs, landmarks