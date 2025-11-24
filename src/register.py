# register placeholder
# src/register.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from src.detector import Detector
from src.embedder import Embedder
from src.utils import USERS_DIR, crop_from_box



def register_user(name, n_images=50, cam_index=0):
    user_dir = os.path.join(USERS_DIR, name)
    imgs_dir = os.path.join(user_dir, 'imgs')
    os.makedirs(imgs_dir, exist_ok=True)

    embedder = Embedder()
    with Detector() as det:
        cap = cv2.VideoCapture(cam_index)
        count = 0
        pbar = tqdm(total=n_images, desc=f'Registering {name}')
        while count < n_images:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, probs, landmarks = det.detect(frame)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                crop = crop_from_box(frame, box, size=embedder.model.input_size if hasattr(embedder.model, 'input_size') else 160)
                if crop is not None:
                    path = os.path.join(imgs_dir, f"{count:03d}.jpg")
                    # crop is RGB already; save BGR
                    cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    count += 1
                    pbar.update(1)
                x1,y1,x2,y2 = [int(round(v)) for v in box]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame, f"Capturing: {count}/{n_images}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.imshow('Register', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        pbar.close()

    # create embeddings
    img_files = sorted([os.path.join(imgs_dir,f) for f in os.listdir(imgs_dir) if f.lower().endswith('.jpg')])
    embs = []
    for p in img_files:
        img = cv2.imread(p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embs.append(embedder.get_embedding(img_rgb))
    if len(embs) == 0:
        raise RuntimeError('No embeddings generated')
    np.save(os.path.join(user_dir, 'embeddings.npy'), np.vstack(embs))
    print('Saved', len(embs), 'embeddings for', name)