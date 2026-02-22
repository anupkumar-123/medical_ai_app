from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

diabetes_model = joblib.load("diabetes.pkl")
heart_model = joblib.load("heart.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    value = [float(x) for x in request.form.values()]
    final = np.array([value])

    if disease == "diabetes":
        prediction = diabetes_model.predict(final)
    else:
        prediction = heart_model.predict(final)
    
    result = "Positive" if prediction[0] == 1 else "Negative"

    return render_template("index.html", result=f"{disease} Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)