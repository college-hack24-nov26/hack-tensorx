import os
import joblib
import numpy as np

CLASSES = {0: "AFib", 1: "NSR", 2: "VTCC"}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "svc_model_V1.h5")

model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")

def predict_signal(signal: np.ndarray, batch: bool = False):
    load_model()

    # batch=True: signal is (n, 2500), returns list of dicts
    # batch=False: signal is (1, 2500), returns single dict
    preds = model.predict(signal)                    # (n,)

    try:
        scores = model.decision_function(signal)     # (n, 3)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs_all = exp_s / exp_s.sum(axis=1, keepdims=True)
    except Exception:
        probs_all = np.zeros((len(preds), 3))
        for i, p in enumerate(preds):
            probs_all[i, int(p)] = 1.0

    results = [
        {
            "prediction": CLASSES[int(preds[i])],
            "probabilities": {CLASSES[j]: round(float(probs_all[i, j]), 4) for j in range(3)}
        }
        for i in range(len(preds))
    ]

    return results if batch else results[0]
