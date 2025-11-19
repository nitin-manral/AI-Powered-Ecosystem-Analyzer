from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ======================================================
# 1. DATA LOAD
# ======================================================
df = pd.read_csv("environment_data.csv")

# date ko string bana ke rakhte hain
df["date"] = df["date"].astype(str)

# ML model load
model = joblib.load("model.pkl")


# ======================================================
# 2. ALERT GENERATOR
# ======================================================
def generate_alerts(dataframe):
    alerts = []

    for _, row in dataframe.iterrows():
        date_str = str(row["date"])

        # ----- Climate / Air ecosystem -----
        if row["temperature"] > 35:
            alerts.append({
                "date": date_str,
                "type": "High Temperature",
                "value": row["temperature"],
                "message": f"High temperature detected: {row['temperature']} °C"
            })

        if row["humidity"] < 30:
            alerts.append({
                "date": date_str,
                "type": "Low Humidity",
                "value": row["humidity"],
                "message": f"Low humidity detected: {row['humidity']} %"
            })

        if row["aqi"] >= 150:
            alerts.append({
                "date": date_str,
                "type": "High AQI",
                "value": row["aqi"],
                "message": f"Unhealthy AQI level: {row['aqi']}"
            })

        # ----- Agricultural soil & crop health -----
        if "soil_moisture" in row and row["soil_moisture"] < 30:
            alerts.append({
                "date": date_str,
                "type": "Low Soil Moisture",
                "value": row["soil_moisture"],
                "message": f"Possible drought / soil dryness: {row['soil_moisture']}%"
            })

        if "crop_health" in row and row["crop_health"] < 70:
            alerts.append({
                "date": date_str,
                "type": "Crop Health Risk",
                "value": row["crop_health"],
                "message": f"Crop health is below normal: index {row['crop_health']}"
            })

        # ----- Forest & wildlife monitoring -----
        if "forest_health" in row and row["forest_health"] < 75:
            alerts.append({
                "date": date_str,
                "type": "Forest Stress",
                "value": row["forest_health"],
                "message": f"Forest vegetation index is low: {row['forest_health']}"
            })

        if "wildlife_index" in row and row["wildlife_index"] < 65:
            alerts.append({
                "date": date_str,
                "type": "Wildlife Activity Low",
                "value": row["wildlife_index"],
                "message": f"Wildlife activity appears reduced: index {row['wildlife_index']}"
            })

        # ----- Water resource & pollution tracking -----
        if "water_quality" in row and row["water_quality"] < 75:
            alerts.append({
                "date": date_str,
                "type": "Water Quality Issue",
                "value": row["water_quality"],
                "message": f"Water quality index is poor: {row['water_quality']}"
            })

    return alerts


# ======================================================
# 3. DAILY REPORTS (list of dict)
# ======================================================
def generate_daily_reports(dataframe, alerts):
    # group by date for averages
    grouped = dataframe.groupby("date").agg({
        "temperature": "mean",
        "humidity": "mean",
        "aqi": ["mean", "max"],
    })
    grouped.columns = ["avg_temp", "avg_humidity", "avg_aqi", "max_aqi"]
    grouped = grouped.reset_index()

    # alert count per date
    alert_count_by_date = {}
    for a in alerts:
        d = a["date"]
        alert_count_by_date[d] = alert_count_by_date.get(d, 0) + 1

    reports = []
    for _, row in grouped.iterrows():
        date_str = row["date"]
        reports.append({
            "date": date_str,
            "avg_temp": round(row["avg_temp"], 2),
            "avg_humidity": round(row["avg_humidity"], 2),
            "avg_aqi": round(row["avg_aqi"], 2),
            "max_aqi": round(row["max_aqi"], 2),
            "alert_count": alert_count_by_date.get(date_str, 0),
        })

    return reports


ALERTS = generate_alerts(df)
REPORTS = generate_daily_reports(df, ALERTS)


# ======================================================
# 4. HOME / DASHBOARD
# ======================================================
@app.route("/")
def home():
    # climate stats
    avg_temp = round(df["temperature"].mean(), 2)
    avg_humidity = round(df["humidity"].mean(), 2)
    avg_aqi = round(df["aqi"].mean(), 2)
    max_aqi = round(df["aqi"].max(), 2)

    latest_temp = df.iloc[-1]["temperature"]
    latest_humidity = df.iloc[-1]["humidity"]
    latest_aqi = df.iloc[-1]["aqi"]

    # ecosystem averages (agar column hai tabhi)
    avg_soil = round(df["soil_moisture"].mean(), 2) if "soil_moisture" in df.columns else 0
    avg_crop = round(df["crop_health"].mean(), 2) if "crop_health" in df.columns else 0
    avg_forest = round(df["forest_health"].mean(), 2) if "forest_health" in df.columns else 0
    avg_wildlife = round(df["wildlife_index"].mean(), 2) if "wildlife_index" in df.columns else 0
    avg_water = round(df["water_quality"].mean(), 2) if "water_quality" in df.columns else 0

    # ----- charts -----
    # AQI
    plt.figure(figsize=(6, 4))
    plt.plot(df["date"], df["aqi"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("AQI Trend Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/aqi_plot.png")
    plt.close()

    # Soil
    if "soil_moisture" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["date"], df["soil_moisture"], marker="o")
        plt.xlabel("Date")
        plt.ylabel("Soil Moisture (%)")
        plt.title("Soil Moisture Trend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/soil_plot.png")
        plt.close()

    # Water
    if "water_quality" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["date"], df["water_quality"], marker="o")
        plt.xlabel("Date")
        plt.ylabel("Water Quality Index")
        plt.title("Water Quality Trend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/water_plot.png")
        plt.close()

    # Forest
    if "forest_health" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["date"], df["forest_health"], marker="o")
        plt.xlabel("Date")
        plt.ylabel("Forest Health Index")
        plt.title("Forest Health Trend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/forest_plot.png")
        plt.close()

    # Wildlife
    if "wildlife_index" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["date"], df["wildlife_index"], marker="o")
        plt.xlabel("Date")
        plt.ylabel("Wildlife Activity Index")
        plt.title("Wildlife Activity Trend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/wildlife_plot.png")
        plt.close()

    table_html = df.head().to_html(classes="table table-striped")

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
        high_aqi_count=high_aqi_count,
        avg_soil_moisture=avg_soil,
        avg_crop_health=avg_crop,
        avg_forest_health=avg_forest,
        avg_wildlife=avg_wildlife,
        avg_water_quality=avg_water,
    )


# ======================================================
# 5. ALERTS PAGE
# ======================================================
@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", alerts=ALERTS)


# ======================================================
# 6. REPORTS PAGE
# ======================================================
@app.route("/reports")
def reports_page():
    return render_template("reports.html", reports=REPORTS)


# ======================================================
# 7. ABOUT PAGE
# ======================================================
@app.route("/about")
def about():
    return render_template("about.html")


# ======================================================
# 8. PREDICT AQI
# ======================================================
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

    return render_template("result.html", aqi=round(predicted_aqi, 2), risk=risk)


if __name__ == "__main__":
    app.run(debug=True)
