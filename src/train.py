# train placeholder
# src/train.py
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from src.utils import USERS_DIR, MODELS_DIR, SVM_PATH, ENC_PATH

os.makedirs(MODELS_DIR, exist_ok=True)


def train_svm():
    X = []
    y = []
    for u in os.listdir(USERS_DIR):
        emb_file = os.path.join(USERS_DIR, u, 'embeddings.npy')
        if not os.path.exists(emb_file):
            continue
        embs = np.load(emb_file)
        for e in embs:
            X.append(e); y.append(u)
    if len(X) == 0:
        raise RuntimeError('No embeddings found. Register users first.')
    X = np.vstack(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y_enc)
    with open(SVM_PATH, 'wb') as f: pickle.dump(clf, f)
    with open(ENC_PATH, 'wb') as f: pickle.dump(le, f)
    print('Saved SVM to', SVM_PATH)