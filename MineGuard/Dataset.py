import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# 1. ZONE DEFINITION
# =========================

zones = [
    {"zone_id": "Z1", "slope": 55, "bench": 16, "rock": "fractured", "x": 80,  "y": 60},
    {"zone_id": "Z2", "slope": 48, "bench": 14, "rock": "medium",    "x": 240, "y": 60},
    {"zone_id": "Z3", "slope": 40, "bench": 12, "rock": "hard",      "x": 400, "y": 60},
    {"zone_id": "Z4", "slope": 52, "bench": 15, "rock": "fractured", "x": 80,  "y": 200},
    {"zone_id": "Z5", "slope": 35, "bench": 10, "rock": "hard",      "x": 240, "y": 200},
]

rock_factor = {
    "hard": 0.5,
    "medium": 1.0,
    "fractured": 1.5
}

# =========================
# 2. SENSOR DATA GENERATION
# =========================

START_TIME = datetime(2025, 1, 1, 0, 0)
MINUTES = 30 * 24 * 60   # 7 days

rows = []
prev_disp = {z["zone_id"]: 0.01 for z in zones}

raining = False
blasting = False

for minute in range(MINUTES):
    time = START_TIME + timedelta(minutes=minute)

    # Weather logic
    if raining:
        rainfall = np.random.uniform(1, 5)
        if np.random.rand() < 0.08:
            raining = False
    else:
        rainfall = 0.0
        if np.random.rand() < 0.03:
            raining = True

    # Blasting logic
    if blasting:
        blast_boost = np.random.uniform(0.3, 0.6)
        if np.random.rand() < 0.1:
            blasting = False
    else:
        blast_boost = 0
        if np.random.rand() < 0.04:
            blasting = True

    for z in zones:
        base_vib = np.random.uniform(0.1, 0.3)
        vibration = base_vib + blast_boost

        slope_factor = z["slope"] / 45
        rf = rock_factor[z["rock"]]

        disp_new = (
            0.01 +
            0.02 * rainfall +
            0.5 * vibration
        ) * rf * slope_factor

        displacement = 0.9 * prev_disp[z["zone_id"]] + disp_new
        prev_disp[z["zone_id"]] = displacement

        rows.append([
            time,
            z["zone_id"],
            vibration,
            rainfall,
            displacement
        ])

sensor_df = pd.DataFrame(
    rows,
    columns=["timestamp", "zone_id", "vibration", "rainfall", "displacement"]
)

sensor_df.to_csv("sensor_data.csv", index=False)

# =========================
# 3. FEATURE EXTRACTION
# =========================

sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
sensor_df = sensor_df.sort_values("timestamp")

WINDOW_HOURS = 3
STEP_HOURS = 1

feature_rows = []

start = sensor_df["timestamp"].min() + timedelta(hours=WINDOW_HOURS)
end = sensor_df["timestamp"].max()

current_time = start

while current_time <= end:
    window_start = current_time - timedelta(hours=WINDOW_HOURS)
    window_df = sensor_df[
        (sensor_df["timestamp"] > window_start) &
        (sensor_df["timestamp"] <= current_time)
    ]

    for z in zones:
        zdf = window_df[window_df["zone_id"] == z["zone_id"]]
        if len(zdf) < 10:
            continue

        vib_mean = zdf["vibration"].mean()
        vib_max = zdf["vibration"].max()
        rain_sum = zdf["rainfall"].sum()

        disp_trend = zdf["displacement"].iloc[-1] - zdf["displacement"].iloc[0]

        # =========================
        # 4. LABELING LOGIC
        # =========================
        risk = int(
            vib_max > 0.75 and
            rain_sum > 6 and
            disp_trend > 0.1
        )

        feature_rows.append([
            current_time,
            z["zone_id"],
            vib_mean,
            vib_max,
            rain_sum,
            disp_trend,
            z["slope"],
            z["bench"],
            z["rock"],
            z["x"],
            z["y"],
            risk
        ])

    current_time += timedelta(hours=STEP_HOURS)

# =========================
# 5. FINAL TRAINING DATASET
# =========================

train_df = pd.DataFrame(
    feature_rows,
    columns=[
        "time", "zone_id",
        "vib_mean_3h", "vib_max_3h",
        "rain_sum_3h", "disp_trend_3h",
        "slope_angle", "bench_height", "rock_type",
        "x_center", "y_center",
        "risk"
    ]
)

train_df.to_csv("training_data.csv", index=False)

print("Synthetic data generation complete.")
print("Files created:")
print(" - sensor_data.csv (raw sensor logs)")
print(" - training_data.csv (ML-ready dataset)")
