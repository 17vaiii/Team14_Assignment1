"""
train.py -- Clean the data, fit baselines, train the network, and save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from app.model import FEATURES, BatteryNet, model_size_info

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--sweep", action="store_true",
                help="train the Mission 2 model-width comparison")
ap.add_argument("--compare-normalization", action="store_true",
                help="compare the selected model with and without input normalization")
ap.add_argument("--size-table", action="store_true",
                help="print the Mission 4 model-size table without training")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

if a.size_table:
    print("width,params,float32_bytes,int8_bytes,float32_SRAM,int8_SRAM,"
          "float32_PSRAM,int8_PSRAM,largest_activation_bytes,MACs")
    for width in (32, 128, 256, 512, 2048):
        sizes = model_size_info(BatteryNet((width, width)))
        print(
            f"{width},{sizes['total_params']},{sizes['total_bytes_f32']},"
            f"{sizes['total_bytes_int8']},"
            f"{sizes['fits_300kb_sram_f32']},{sizes['fits_300kb_sram_int8']},"
            f"{sizes['total_bytes_f32'] <= 8 * 1024 * 1024},"
            f"{sizes['total_bytes_int8'] <= 8 * 1024 * 1024},"
            f"{width * 4},{sizes['total_macs']}"
        )
    sram_limit_bytes = 300 * 1024
    float32_width = next(
        width for width in range(1, 3000)
        if 4 * (width * width + 7 * width + 1) > sram_limit_bytes
    )
    int8_width = next(
        width for width in range(1, 3000)
        if width * width + 7 * width + 1 > sram_limit_bytes
    )
    print(f"float32_first_exceeds_300KiB,width={float32_width}")
    print(f"int8_first_exceeds_300KiB,width={int8_width}")
    raise SystemExit(0)


def fit_model(widths, x_train, y_train, x_val, y_val, epochs):
    torch.manual_seed(0)
    model = BatteryNet(widths)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    xt, yt = torch.tensor(x_train), torch.tensor(y_train)
    xvt = torch.tensor(x_val)

    for _ in range(epochs):
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), 128):
            batch = order[start:start + 128]
            loss = torch.nn.functional.mse_loss(model(xt[batch]), yt[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_mae = float(torch.mean(torch.abs(model(xt) - yt)))
        val_mae = float(torch.mean(torch.abs(model(xvt) - torch.tensor(y_val))))
    return model, train_mae, val_mae


df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")

df[["battery_v", "temp_c", "rings_per_day",
    "wifi_rssi_dbm", "age_months", "days_left"]].hist(
    bins=30,
    figsize=(12, 8),
)

plt.tight_layout()
plt.savefig(f"team{a.team}_histograms.png", dpi=160)
plt.close()

for feature in FEATURES:
    plt.scatter(df[feature], df["days_left"], s=8, alpha=0.35)
    plt.xlabel(feature)
    plt.ylabel("days_left")
    plt.title(f"Team {a.team}: days_left vs {feature}")
    plt.tight_layout()
    plt.savefig(f"team{a.team}_days_vs_{feature}.png", dpi=160)
    plt.close()

# Team 14 sensor glitches found during EDA.
bad = (
    (df["battery_v"] <= 0)
    | (df["temp_c"] < -25)
    | (df["rings_per_day"] > 40)
    | (df["wifi_rssi_dbm"] > 0)
    | (df["age_months"] < 0)
    | (df["days_left"] < 0)
).to_numpy()
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# Fit scaling values on training rows only to avoid leaking validation data.
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd = np.where(sd == 0, 1.0, sd).astype(np.float32)
Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")
linear_X = np.column_stack((X, np.ones(len(X), np.float32)))
linear_coefficients = np.linalg.lstsq(linear_X, y, rcond=None)[0]
linear_Xv = np.column_stack((Xv, np.ones(len(Xv), np.float32)))
linear_prediction = linear_Xv @ linear_coefficients
linear_mae = mae(linear_prediction, yv)
print(f"baseline (linear regression):  {linear_mae:.2f} days")

plt.scatter(yv, linear_prediction, s=8, alpha=0.35)
limits = [0, max(float(yv.max()), float(linear_prediction.max()))]
plt.plot(limits, limits, "r--", label="perfect prediction")
plt.xlabel("True days_left")
plt.ylabel("Linear prediction")
plt.title(f"Team {a.team}: linear baseline predicted vs true")
plt.legend()
plt.tight_layout()
plt.savefig(f"team{a.team}_linear_predicted_vs_true.png", dpi=160)
plt.close()

if a.compare_normalization:
    _, raw_train_mae, raw_val_mae = fit_model(hidden, X, y, Xv, yv, a.epochs)
    print(f"without normalisation: training MAE {raw_train_mae:.2f}, "
          f"validation MAE {raw_val_mae:.2f} days")

net = None
train_mae = val_mae = None
if a.sweep:
    comparison = []
    for widths in ((4,), (8,), (16,), (32,), (64,),
                   (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)):
        candidate, candidate_train_mae, candidate_val_mae = fit_model(
            widths, Xn, y, Xvn, yv, a.epochs
        )
        params = sum(parameter.numel() for parameter in candidate.parameters())
        label = ",".join(map(str, widths))
        comparison.append((label, params, candidate_val_mae))
        print(f"hidden {label:5s} parameters {params:5d} "
              f"training MAE {candidate_train_mae:.2f} "
              f"validation MAE {candidate_val_mae:.2f} days")
        if widths == hidden:
            net = candidate
            train_mae, val_mae = candidate_train_mae, candidate_val_mae

    plt.figure(figsize=(8, 5))
    plt.scatter([row[1] for row in comparison], [row[2] for row in comparison])
    for label, params, error in comparison:
        plt.annotate(label, (params, error), xytext=(5, 4), textcoords="offset points")
    plt.xlabel("Number of parameters")
    plt.ylabel("Validation MAE (days)")
    plt.title(f"Team {a.team}: validation MAE vs model size")
    plt.tight_layout()
    plt.savefig("mission2_model_comparison.png", dpi=160)
    plt.close()

if net is None:
    net, train_mae, val_mae = fit_model(hidden, Xn, y, Xvn, yv, a.epochs)

n_params = sum(p.numel() for p in net.parameters())
print(f"parameters: {n_params}")
print(f"training MAE:   {train_mae:.2f} days")
print(f"validation MAE: {val_mae:.2f} days")
os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")