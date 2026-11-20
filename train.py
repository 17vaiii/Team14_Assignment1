"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
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
# TODO Mission 2: normalisation. Why must mu and sd come from the TRAINING rows only?
mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)

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
plt.title("Team 14: linear baseline predicted vs true")
plt.legend()
plt.tight_layout()
plt.savefig("team14_linear_predicted_vs_true.png", dpi=160)
plt.close()

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}:            {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
