import numpy as np
import joblib
from config import MODEL_PATH
from llm_feature_extractor import extract_fincrime_features
from embedding_generator import get_embedding

model = joblib.load(MODEL_PATH)

CLASSES = {
    0: "Normal",
    1: "Suspicious",
    2: "Likely Financial Crime"
}

def predict_case(narrative, amount, age):

    llm = extract_fincrime_features(narrative)
    emb = get_embedding(narrative)

    vector = [amount, age]
    vector += list(llm.values())
    vector += list(emb)

    X = np.array(vector).reshape(1, -1)

    proba = model.predict_proba(X)[0]
    label = model.predict(X)[0]

    return CLASSES[label], proba


# Example
case = """
Customer opened a new account.
Received transfers from 6 unrelated individuals.
Funds withdrawn immediately via ATM.
"""

label, prob = predict_case(case, 9800, 19)

print("Prediction:", label)
print("Probabilities:", prob)
