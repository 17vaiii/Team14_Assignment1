
"""
app/main.py -- battery prediction API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> health and model information
GET  /model_info      -> per-layer params, bytes, MACs, doorbell fit
POST /predict         -> one battery reading
POST /predict_batch   -> multiple battery readings

Out-of-range readings return HTTP 422.
"""

import json
import time

import numpy as np
import torch

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from app.model import FEATURES, BatteryNet, model_size_info


# --------------------------------------------------
# 1. Load model and configuration
# --------------------------------------------------

cfg = json.load(open("model/config.json"))

net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(
    torch.load("model/model.pt", weights_only=True)
)

net.eval()
torch.set_num_threads(1)

MU = np.array(cfg["mu"], np.float32)
SD = np.array(cfg["sd"], np.float32)

N_PARAMS = sum(p.numel() for p in net.parameters())


# --------------------------------------------------
# 2. Input validation
# --------------------------------------------------

class Reading(BaseModel):

    model_config = ConfigDict(
        allow_inf_nan=False
    )

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean_values(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(
                "boolean values are not valid readings"
            )
        return value

    battery_v: float = Field(
        ge=2.0,
        le=4.5,
        allow_inf_nan=False
    )

    temp_c: float = Field(
        ge=-40.0,
        le=85.0,
        allow_inf_nan=False
    )

    rings_per_day: float = Field(
        ge=0.0,
        le=100.0,
        allow_inf_nan=False
    )

    wifi_rssi_dbm: float = Field(
        ge=-100.0,
        le=0.0,
        allow_inf_nan=False
    )

    age_months: float = Field(
        ge=0.0,
        le=360.0,
        allow_inf_nan=False
    )


class Batch(BaseModel):
    rows: list[Reading] = Field(
        min_length=1,
        max_length=10000
    )


# --------------------------------------------------
# 3. OpenAPI schemas
# --------------------------------------------------

READING_OPENAPI_SCHEMA = Reading.model_json_schema()

BATCH_OPENAPI_SCHEMA = {
    "title": "Batch",
    "type": "object",
    "properties": {
        "rows": {
            "title": "Rows",
            "type": "array",
            "items": READING_OPENAPI_SCHEMA,
            "minItems": 1,
            "maxItems": 10000,
        }
    },
    "required": ["rows"],
}


# --------------------------------------------------
# 4. FastAPI initialization
# --------------------------------------------------

app = FastAPI(title="Doorbell battery API")


def ms(t0):
    return round(
        (time.perf_counter() - t0) * 1000,
        3
    )


# --------------------------------------------------
# 5. Model information endpoint
# --------------------------------------------------

@app.get("/model_info")
def model_info():
    return model_size_info(net)


# --------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "team": cfg["team"],
        "params": N_PARAMS,
        "val_mae": cfg["val_mae"],
    }


# --------------------------------------------------
# 7. Shared prediction pipeline
# --------------------------------------------------

async def run(request: Request, schema):

    t_all = time.perf_counter()
    t = {}

    # Stage 1: Parse and validate input

    t0 = time.perf_counter()

    try:
        data = schema.model_validate_json(
            await request.body()
        )

    except ValidationError as e:

        return JSONResponse(
            status_code=422,
            content={
                "error": "bad reading",
                "detail": json.loads(
                    e.json(
                        include_url=False,
                        include_context=False,
                        include_input=False
                    )
                ),
            },
        )

    t["parse"] = ms(t0)

    rows = (
        data.rows
        if isinstance(data, Batch)
        else [data]
    )

    # Stage 2: Normalisation

    t0 = time.perf_counter()

    x = (
        np.array(
            [
                [getattr(r, f) for f in FEATURES]
                for r in rows
            ],
            dtype=np.float32,
        ) - MU
    ) / SD

    t["normalise"] = ms(t0)

    # Stage 3: Model inference

    t0 = time.perf_counter()

    with torch.no_grad():
        y = net(
            torch.from_numpy(x)
        ).numpy()

    t["inference"] = ms(t0)

    # Stage 4: Build API response

    t0 = time.perf_counter()

    days = [
        round(float(v), 2)
        for v in np.maximum(y, 0)
    ]

    body = {
        "days_left": (
            days
            if isinstance(data, Batch)
            else days[0]
        )
    }

    t["build"] = ms(t0)

    # Total server processing time

    t["server_total"] = ms(t_all)

    body["timings_ms"] = t

    return body


# --------------------------------------------------
# 8. Single prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": READING_OPENAPI_SCHEMA
                }
            },
        }
    },
)
async def predict(request: Request):
    return await run(request, Reading)


# --------------------------------------------------
# 9. Batch prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict_batch",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": BATCH_OPENAPI_SCHEMA
                }
            },
        }
    },
)
async def predict_batch(request: Request):
    return await run(request, Batch)
