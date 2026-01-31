import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("training_data.csv")

# Convert time to datetime (important for sorting)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

# =========================
# 2. SELECT FEATURES
# =========================

FEATURE_COLS = [
    "vib_mean_3h",
    "vib_max_3h",
    "rain_sum_3h",
    "disp_trend_3h",
    "slope_angle",
    "bench_height",
    "rock_type"
]

TARGET_COL = "risk"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# =========================
# 3. ENCODE CATEGORICAL DATA
# =========================

rock_encoder = LabelEncoder()
X["rock_type"] = rock_encoder.fit_transform(X["rock_type"])

# Save encoder for later use
joblib.dump(rock_encoder, "rock_encoder.pkl")

# =========================
# 4. TRAIN / TEST SPLIT (TIME-AWARE)
# =========================

# Use first 80% of time for training, last 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# =========================
# 5. TRAIN RANDOM FOREST MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 6. EVALUATION
# =========================

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 7. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(importance)

# =========================
# 8. SAVE MODEL
# =========================

joblib.dump(model, "rockfall_risk_model.pkl")

print("\nModel training complete.")
print("Saved files:")
print("- rockfall_risk_model.pkl")
print("- rock_encoder.pkl")
