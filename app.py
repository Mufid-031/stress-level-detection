from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("model/model-tanpa-smote.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return jsonify({"message": "API Model Stress Aktif"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([
            data["x"],
            data["y"],
            data["z"],
            data["bvp"],
            data["eda"],
            data["hr"]
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        label = int(prediction[0])

        label_meaning = {
            0: "No Stress 😊",
            1: "Low Stress 😌",
            2: "Medium Stress 😐"
        }

        return jsonify({
            "status": "success",
            "prediction": label,
            "label_text": label_meaning[label]
        })
 
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        X = df[["x", "y", "z", "bvp", "eda", "hr"]]
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)

        label_meaning = {
            0: "No Stress 😊",
            1: "Low Stress 😌",
            2: "Medium Stress 😐"
        }

        return jsonify({
            "status": "success",
            "predictions": predictions.tolist(),
            "labels": [label_meaning[pred] for pred in predictions]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)