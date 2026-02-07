import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from datetime import timedelta

# =========================
# CONFIG
# =========================

WINDOW_HOURS = 3
STEP_HOURS = 1
PAUSE_SECONDS = 1.0   # visualization speed

MODEL_FILE = "rockfall_risk_model.pkl"
ENCODER_FILE = "rock_encoder.pkl"
SENSOR_FILE = "sensor_data.csv"

# =========================
# LOAD
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, MODEL_FILE))
encoder = joblib.load(os.path.join(BASE_DIR, ENCODER_FILE))

sensor_df = pd.read_csv(os.path.join(BASE_DIR, SENSOR_FILE))
sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
sensor_df = sensor_df.sort_values("timestamp")

# =========================
# ZONES
# =========================

zones = [
    {"zone_id": "Z1", "slope": 55, "bench": 16, "rock": "fractured", "x": 80,  "y": 60},
    {"zone_id": "Z2", "slope": 48, "bench": 14, "rock": "medium",    "x": 240, "y": 60},
    {"zone_id": "Z3", "slope": 40, "bench": 12, "rock": "hard",      "x": 400, "y": 60},
    {"zone_id": "Z4", "slope": 52, "bench": 15, "rock": "fractured", "x": 80,  "y": 200},
    {"zone_id": "Z5", "slope": 35, "bench": 10, "rock": "hard",      "x": 240, "y": 200},
]

# =========================
# TIME SETUP
# =========================

current_time = sensor_df["timestamp"].min() + timedelta(hours=WINDOW_HOURS)
end_time = sensor_df["timestamp"].max()

# =========================
# FIGURE SETUP
# =========================

fig, ax = plt.subplots(figsize=(9, 7))
plt.show(block=False)   # 🔴 important

# =========================
# LOOP
# =========================

while current_time <= end_time:

    history_df = sensor_df[sensor_df["timestamp"] <= current_time]
    window_df = history_df[
        history_df["timestamp"] > current_time - timedelta(hours=WINDOW_HOURS)
    ]

    features = []

    for z in zones:
        zdf = window_df[window_df["zone_id"] == z["zone_id"]]
        if len(zdf) < 10:
            continue

        features.append({
            "zone_id": z["zone_id"],
            "vib_mean_3h": zdf["vibration"].mean(),
            "vib_max_3h": zdf["vibration"].max(),
            "rain_sum_3h": zdf["rainfall"].sum(),
            "disp_trend_3h": zdf["displacement"].iloc[-1] - zdf["displacement"].iloc[0],
            "slope_angle": z["slope"],
            "bench_height": z["bench"],
            "rock_type": z["rock"],
            "x": z["x"],
            "y": z["y"]
        })

    if not features:
        current_time += timedelta(hours=STEP_HOURS)
        continue

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
    feat_df["risk_prob"] = model.predict_proba(X)[:, 1]

    # =========================
    # DRAW
    # =========================

    ax.clear()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 350)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.set_title(f"Rockfall Risk Map\nTime: {current_time}")

    for _, row in feat_df.iterrows():
        color = cm.Reds(row["risk_prob"]) if row["risk"] == 1 else cm.Greens(0.4)
        edge = "darkred" if row["risk"] == 1 else "darkgreen"

        rect = patches.Rectangle(
            (row["x"] - 30, row["y"] - 30),
            60,
            60,
            edgecolor=edge,
            facecolor=color,
            linewidth=2,
            alpha=0.85
        )
        ax.add_patch(rect)

        ax.text(
            row["x"], row["y"],
            f"{row['zone_id']}\n{row['risk_prob']:.2f}",
            ha="center", va="center", fontsize=10, fontweight="bold"
        )

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(PAUSE_SECONDS)

    current_time += timedelta(hours=STEP_HOURS)

plt.show()
