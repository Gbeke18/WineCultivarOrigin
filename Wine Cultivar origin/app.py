from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("wine_cultivar_model.pkl", "rb") as f:
    model, scaler, features = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        input_data = [float(request.form[feature]) for feature in features]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", features=features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
