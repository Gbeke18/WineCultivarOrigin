from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "wine_cultivar_model.pkl")

# Load model bundle
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            input_data = [float(request.form[f]) for f in features]
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            pred = model.predict(input_scaled)[0]
            prediction = f"Cultivar {pred + 1}"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        features=features,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run()
