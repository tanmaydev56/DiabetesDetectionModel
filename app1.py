
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib, json
from flask import Flask, request, jsonify
import numpy as np

# -------------------------------------------------
# 1. Bring in the same custom layer you trained with
# -------------------------------------------------
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.tanh_dense   = tf.keras.layers.Dense(n_features, activation='tanh')
        self.softmax_dense = tf.keras.layers.Dense(n_features, activation='softmax')
        super().build(input_shape)

    def call(self, inputs):
        attention = self.tanh_dense(inputs)
        attention = self.softmax_dense(attention)
        return inputs * attention

    def get_config(self):
        return super().get_config()
app = Flask(__name__)

# ------------------------------------------------------------------
# 1. Load artefacts once at start-up
# ------------------------------------------------------------------
MODEL_PATH  = 'diabetes_model.h5'
SCALER_PATH = 'age_scaler.pkl'

model  = load_model(MODEL_PATH, custom_objects={'FocalLoss': None})   # Attention layer is already serialised
scaler = joblib.load(SCALER_PATH)

# Column order in the training data (drop class)
COLS = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
    'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
    'Itching', 'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

# ------------------------------------------------------------------
# 2. Helper utilities
# ------------------------------------------------------------------
def encode_binary(x: str) -> int:
    """Convert Yes/No or Male/Female to 1/0."""
    val = str(x).strip().lower()
    if val in ("yes", "male"):
        return 1
    elif val in ("no", "female"):
        return 0
    raise ValueError(f"Unrecognised value: {x}")

def preprocess(payload: dict) -> np.ndarray:
    """
    Turn the raw JSON into a numpy array in the exact order and scale
    the model expects.
    """
    row = []
    # Age first
    age_raw = float(payload["Age"])
    age_scaled = scaler.transform([[age_raw]])[0][0]
    row.append(age_scaled)

    # Remaining categorical features
    for col in COLS[1:]:
        row.append(encode_binary(payload[col]))
    return np.array([row], dtype=np.float32)

# ------------------------------------------------------------------
# 3. REST endpoints
# ------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON (example):
    {
      "Age": 45,
      "Gender": "Male",
      "Polyuria": "Yes",
      ...
    }
    """
    try:
        data = request.get_json(force=True)
        X = preprocess(data)
        prob = float(model.predict(X)[0][0])
        label = "Positive" if prob >= 0.5 else "Negative"
        return jsonify({"probability": prob, "class": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/columns", methods=["GET"])
def columns():
    """Return the ordered list of feature names."""
    return jsonify(COLS)

# ------------------------------------------------------------------
# 4. Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)