"""
app/main.py -- battery prediction API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet, model_size_info

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    battery_v: float = Field(ge=2.0, le=4.5, allow_inf_nan=False)
    temp_c: float = Field(ge=-40.0, le=85.0, allow_inf_nan=False)
    rings_per_day: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    wifi_rssi_dbm: float = Field(ge=-100.0, le=0.0, allow_inf_nan=False)
    age_months: float = Field(ge=0.0, le=360.0, allow_inf_nan=False)


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)

@app.get("/model_info")
def model_info():
    return model_size_info(net)


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    t0_normalise = time.perf_counter()
    x = (
        np.array(
            [[getattr(r, f) for f in FEATURES] for r in rows],
            dtype=np.float32,
        )
        - MU
    ) / SD
    t["normalise"] = ms(t0_normalise)

    t0_inference = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0_inference)

    t0_build = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0_build)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": Reading.model_json_schema()
                }
            }
        }
    }
)
async def predict(request: Request):
    return await run(request, Reading)


@app.post(
    "/predict_batch",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": Batch.model_json_schema()
                }
            }
        }
    }
)
async def predict_batch(request: Request):
    return await run(request, Batch)
