import pandas as pd

df = pd.read_csv("environment_data.csv")
df["date"] = pd.to_datetime(df["date"])

# sirf year ko 2025 kar rahe hain
df["date"] = df["date"].dt.strftime("2025-%m-%d")

df.to_csv("environment_data.csv", index=False)
print("Dates updated to 2025.")
