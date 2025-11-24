# embedder placeholder
# embedder placeholder
# src/embedder.py
# FaceNet embeddings using facenet-pytorch InceptionResnetV1

from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np

class Embedder:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding(self, face_rgb):
        """Compute a 512-d embedding for a face.
        face_rgb: HxWx3 numpy array in RGB [0..255]
        returns: 1D numpy array (512,) L2-normalized
        """
        img = face_rgb.astype('float32') / 255.0
        tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        emb = emb.cpu().numpy().reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb