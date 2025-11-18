import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Dataset load karna
df = pd.read_csv("environment_data.csv")

print("Columns in dataset:", df.columns.tolist())
print("\nFirst 5 rows of data:")
print(df.head())

# 2. Features (X) and Target (y)
# Hum AQI ko predict kar rahe hain based on temperature, humidity, pm25, pm10
X = df[["temperature", "humidity", "pm25", "pm10"]]
y = df["aqi"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model define + train
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# 6. Model save karna
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
