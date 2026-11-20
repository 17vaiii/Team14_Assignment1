"""
Compare 400 single predictions with one batch prediction.

Run from the repository root:
    python client/compare_batch.py --url http://localhost:8000
"""
import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from bench import reading


READING_COUNT = 400


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    url = args.url.rstrip("/")

    rng = np.random.default_rng(1)
    readings = [reading(rng) for _ in range(READING_COUNT)]
    run_timestamp_utc = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    with requests.Session() as session:
        single_start = time.perf_counter()
        single_predictions = []
        for item in readings:
            response = session.post(url + "/predict", json=item, timeout=30)
            response.raise_for_status()
            single_predictions.append(response.json()["days_left"])
        single_total_ms = (time.perf_counter() - single_start) * 1000

        batch_start = time.perf_counter()
        response = session.post(url + "/predict_batch", json={"rows": readings}, timeout=30)
        response.raise_for_status()
        batch_predictions = response.json()["days_left"]
        batch_total_ms = (time.perf_counter() - batch_start) * 1000

    if not isinstance(batch_predictions, list):
        raise ValueError("/predict_batch response 'days_left' must be a list")

    differences = [
        abs(single - batch)
        for single, batch in zip(single_predictions, batch_predictions)
    ]
    predictions_match = (
        len(batch_predictions) == READING_COUNT
        and all(difference == 0 for difference in differences)
    )
    max_abs_difference = max(differences) if differences else ""
    speedup = single_total_ms / batch_total_ms

    result = {
        "run_timestamp_utc": run_timestamp_utc,
        "url": url,
        "reading_count": READING_COUNT,
        "single_requests": READING_COUNT,
        "single_total_ms": round(single_total_ms, 3),
        "batch_requests": 1,
        "batch_total_ms": round(batch_total_ms, 3),
        "speedup": round(speedup, 3),
        "single_predictions": len(single_predictions),
        "batch_predictions": len(batch_predictions),
        "predictions_match": predictions_match,
        "max_abs_difference": max_abs_difference,
    }

    output_path = Path(__file__).with_name(
        f"compare_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with output_path.open("w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=result)
        writer.writeheader()
        writer.writerow(result)

    print(f"400 single predictions: {single_total_ms:.3f} ms")
    print(f"one batch prediction: {batch_total_ms:.3f} ms")
    print(f"batch speedup: {speedup:.3f}x")
    print(f"predictions match: {predictions_match}")
    if max_abs_difference != "":
        print(f"maximum absolute difference: {max_abs_difference}")
    print(f"results saved to {output_path}")


if __name__ == "__main__":
    main()
