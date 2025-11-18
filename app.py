from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load dataset & model
df = pd.read_csv("environment_data.csv")
model = joblib.load("model.pkl")


# ============================
#  HELPER: Generate Alerts
# ============================
def generate_alerts(dataframe):
    alerts = []

    for _, row in dataframe.iterrows():
        date_str = str(row["date"])

        # High temperature
        if row["temperature"] > 35:
            alerts.append({
                "date": date_str,
                "type": "High Temperature",
                "value": row["temperature"],
                "message": f"High temperature detected: {row['temperature']} °C"
            })

        # Low humidity
        if row["humidity"] < 30:
            alerts.append({
                "date": date_str,
                "type": "Low Humidity",
                "value": row["humidity"],
                "message": f"Low humidity detected: {row['humidity']} %"
            })

        # High AQI
        if row["aqi"] >= 150:
            alerts.append({
                "date": date_str,
                "type": "High AQI",
                "value": row["aqi"],
                "message": f"Unhealthy AQI level: {row['aqi']}"
            })

    return alerts


# Alerts list once (data static hai)
ALERTS = generate_alerts(df)


# ============================
#  HOME / DASHBOARD PAGE
# ============================
@app.route("/")
def home():

    avg_temp = round(df["temperature"].mean(), 2)
    avg_humidity = round(df["humidity"].mean(), 2)
    avg_aqi = round(df["aqi"].mean(), 2)
    max_aqi = round(df["aqi"].max(), 2)

    latest_temp = df.iloc[-1]["temperature"]
    latest_humidity = df.iloc[-1]["humidity"]
    latest_aqi = df.iloc[-1]["aqi"]

    # AQI Trend chart
    plt.figure(figsize=(6, 4))
    plt.plot(df["date"], df["aqi"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("AQI Trend Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/aqi_plot.png")
    plt.close()

    # Table HTML
    table_html = df.head().to_html(classes="table table-striped")

    # Alert counts (dashboard pe show karne ke liye)
    high_temp_count = sum(1 for a in ALERTS if a["type"] == "High Temperature")
    low_hum_count = sum(1 for a in ALERTS if a["type"] == "Low Humidity")
    high_aqi_count = sum(1 for a in ALERTS if a["type"] == "High AQI")

    return render_template(
        "index.html",
        avg_temp=avg_temp,
        avg_humidity=avg_humidity,
        avg_aqi=avg_aqi,
        max_aqi=max_aqi,
        latest_temp=latest_temp,
        latest_humidity=latest_humidity,
        latest_aqi=latest_aqi,
        table_html=table_html,
        high_temp_count=high_temp_count,
        low_hum_count=low_hum_count,
        high_aqi_count=high_aqi_count
    )


# ============================
#  ALERTS PAGE
# ============================
@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", alerts=ALERTS)


# ============================
#  PREDICTION ROUTE
# ============================
@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temperature"])
    hum = float(request.form["humidity"])
    pm25 = float(request.form["pm25"])
    pm10 = float(request.form["pm10"])

    input_data = [[temp, hum, pm25, pm10]]
    predicted_aqi = model.predict(input_data)[0]

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
