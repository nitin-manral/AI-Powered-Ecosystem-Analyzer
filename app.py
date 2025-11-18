from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # For backend rendering
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load dataset and model
df = pd.read_csv("environment_data.csv")
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html", tables=[df.head().to_html(classes="table table-striped")], titles=df.columns.values)


@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temperature"])
    hum = float(request.form["humidity"])
    pm25 = float(request.form["pm25"])
    pm10 = float(request.form["pm10"])

    input_data = [[temp, hum, pm25, pm10]]
    predicted_aqi = model.predict(input_data)[0]

    # Risk level
    if predicted_aqi <= 50:
        risk = "GOOD 😊"
    elif predicted_aqi <= 100:
        risk = "MODERATE 🙂"
    elif predicted_aqi <= 200:
        risk = "UNHEALTHY 😷"
    else:
        risk = "HAZARDOUS ☠️"

    return render_template("result.html", aqi=predicted_aqi, risk=risk)


if __name__ == "__main__":
    app.run(debug=True)
