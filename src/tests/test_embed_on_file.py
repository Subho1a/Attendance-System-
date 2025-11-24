import cv2
from src.detector import Detector
from src.embedder import Embedder

IMG_PATH = r"C:\Users\arind\COADING\AI_ML\attendance_system\subho.png"

def test():
    print("Loading test image...")
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {IMG_PATH}")

    print("Running MTCNN detection...")
    with Detector() as det:
        boxes, probs, landmarks = det.detect(img)
        if boxes is None or len(boxes) == 0:
            raise RuntimeError("No face detected in test image")
        box = boxes[0]
        x1, y1, x2, y2 = [int(v) for v in box]
        face = img[y1:y2, x1:x2]

    print("Running FaceNet embedder...")
    embedder = Embedder()
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    emb = embedder.get_embedding(face_rgb)

    print("Embedding shape:", emb.shape)
    print("Embedding first 10:", emb[:10])
    print("SUCCESS!")

if __name__ == "__main__":
    test()
