import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# =========================
# CONFIG
# =========================

WINDOW_HOURS = 3
STEP_HOURS = 1

MODEL_FILE = "rockfall_risk_model.pkl"
ENCODER_FILE = "rock_encoder.pkl"
SENSOR_FILE = "sensor_data.csv"

# =========================
# LOAD PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, MODEL_FILE))
encoder = joblib.load(os.path.join(BASE_DIR, ENCODER_FILE))

sensor_df = pd.read_csv(os.path.join(BASE_DIR, SENSOR_FILE))
sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
sensor_df = sensor_df.sort_values("timestamp")

# =========================
# ZONE METADATA
# =========================

zones = [
    {"zone_id": "Z1", "slope": 55, "bench": 16, "rock": "fractured", "x": 80,  "y": 60},
    {"zone_id": "Z2", "slope": 48, "bench": 14, "rock": "medium",    "x": 240, "y": 60},
    {"zone_id": "Z3", "slope": 40, "bench": 12, "rock": "hard",      "x": 400, "y": 60},
    {"zone_id": "Z4", "slope": 52, "bench": 15, "rock": "fractured", "x": 80,  "y": 200},
    {"zone_id": "Z5", "slope": 35, "bench": 10, "rock": "hard",      "x": 240, "y": 200},
]

# =========================
# LIVE SIMULATION LOOP
# =========================

start_time = sensor_df["timestamp"].min() + timedelta(hours=WINDOW_HOURS)
end_time = sensor_df["timestamp"].max()

plt.ion()  # interactive mode
fig, ax = plt.subplots(figsize=(8, 6))

current_time = start_time

while current_time <= end_time:

    # Take all data up to now
    history_df = sensor_df[sensor_df["timestamp"] <= current_time]

    # Slice last window
    window_start = current_time - timedelta(hours=WINDOW_HOURS)
    window_df = history_df[history_df["timestamp"] > window_start]

    features = []

    for z in zones:
        zdf = window_df[window_df["zone_id"] == z["zone_id"]]
        if len(zdf) < 10:
            continue

        vib_mean = zdf["vibration"].mean()
        vib_max = zdf["vibration"].max()
        rain_sum = zdf["rainfall"].sum()
        disp_trend = zdf["displacement"].iloc[-1] - zdf["displacement"].iloc[0]

        features.append({
            "zone_id": z["zone_id"],
            "vib_mean_3h": vib_mean,
            "vib_max_3h": vib_max,
            "rain_sum_3h": rain_sum,
            "disp_trend_3h": disp_trend,
            "slope_angle": z["slope"],
            "bench_height": z["bench"],
            "rock_type": z["rock"],
            "x": z["x"],
            "y": z["y"]
        })

    feat_df = pd.DataFrame(features)
    feat_df["rock_type"] = encoder.transform(feat_df["rock_type"])

    X = feat_df[
        [
            "vib_mean_3h",
            "vib_max_3h",
            "rain_sum_3h",
            "disp_trend_3h",
            "slope_angle",
            "bench_height",
            "rock_type"
        ]
    ]

    feat_df["risk"] = model.predict(X)

    # =========================
    # PLOT REFRESH
    # =========================

    ax.clear()
    ax.set_title(f"Live Rockfall Risk Map\nTime: {current_time}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    for _, row in feat_df.iterrows():
        if row["risk"] == 1:
            ax.scatter(row["x"], row["y"], s=300, marker="X")
        else:
            ax.scatter(row["x"], row["y"], s=200)

        ax.text(row["x"] + 5, row["y"] + 5, row["zone_id"])

    plt.pause(1.5)  # controls animation speed
    current_time += timedelta(hours=STEP_HOURS)

plt.ioff()
plt.show()
