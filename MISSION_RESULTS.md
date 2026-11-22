# Team 14 Mission Results

These results were measured from Team 14's own training data and the completed API.

## Mission 1: Meet the data and set the bar

- Sensor rows: 2000; glitch rows removed: 16.
- The mean-baseline validation MAE is 62.01 days.
- The linear-regression validation MAE is 26.98 days.
- The starter-style network without input normalisation has validation MAE 17.24 days.
- Histograms and input-versus-output plots are saved as `team14_histograms.png` and
  `team14_days_vs_<feature>.png`.
- The linear predicted-versus-true plot is saved as
  `team14_linear_predicted_vs_true.png`.

## Mission 2: Network size and normalisation

- Normalised input scaling was computed from training rows only.
- The same `32,32` architecture measured:
  - Training MAE 15.29 days and validation MAE 17.24 days without normalisation.
  - Training MAE 4.82 days and validation MAE 6.11 days with normalisation.
- Width sweep validation MAE:

| Hidden architecture | Parameters | Training MAE | Validation MAE |
|---|---:|---:|---:|
| 4 | 29 | 16.10 | 15.34 |
| 8 | 57 | 20.17 | 21.16 |
| 16 | 113 | 12.49 | 13.00 |
| 32 | 225 | 8.16 | 9.45 |
| 64 | 449 | 20.07 | 21.18 |
| 4,4 | 49 | 10.67 | 10.94 |
| 8,8 | 129 | 7.03 | 8.33 |
| 16,16 | 385 | 5.92 | 7.89 |
| 32,32 | 1281 | 4.82 | 6.11 |
| 64,64 | 4609 | 4.16 | 6.22 |

The `32,32` model is the selected size: it is much more accurate than smaller models,
has only a 1.29-day training/validation gap, and uses far fewer parameters than `64,64`
without materially worse validation MAE. The sweep plot is
`mission2_model_comparison.png`. The expected MAE for 400 unseen readings is 6.11 days,
based on the validation MAE.

## Mission 3: API and latency

All five sensor ranges reject impossible readings with HTTP 422. The API times `parse`,
`normalise`, `inference`, `build`, and `server_total` for both single and batch requests.

The LLM prediction logged before measurement was that model inference would be the
slowest server stage and that the laptop-to-Codespace round trip would dominate the
total request. The local measurement did not make inference dominate because localhost
overhead and framework/network handling were larger than the sub-millisecond server stages;
inference was still the largest timed server stage.

Local keepalive benchmark with 50 requests:

- First request: 3.1 ms round trip.
- Median round trip: 1.364 ms.
- Median server total: 0.176 ms.
- Median stages: parse 0.029 ms, normalise 0.017 ms, inference 0.086 ms, build 0.011 ms.
- Median network/framework share: 1.225 ms.

Local no-keepalive benchmark with 50 requests:

- First request: 2.0 ms round trip.
- Median round trip: 1.746 ms.
- Median server total: 0.154 ms.
- Median network/framework share: 1.605 ms.

The no-keepalive run is slower overall because connection setup occurs repeatedly.
Localhost avoids external network and laptop-to-Codespace delay; the benchmark measures
server and framework overhead directly instead.

One measured batch of 400 rows took 6.916 ms end-to-end, while 400 individual requests
took 753.542 ms. The batch was 108.96x faster, and every batch prediction exactly matched
its corresponding single prediction. The improvement comes from reusing one HTTP request,
one parse, and one batched tensor/inference call.

## Mission 4: Doorbell memory fit

The selected `32,32` model has 1281 parameters:

- Float32 weights and biases: 5124 bytes.
- Int8 weights and biases: 1281 bytes.
- Largest single-prediction activation: 128 bytes (two 32-wide float32 hidden activations).
- MACs per prediction: 1216.
- It fits in 300 KiB SRAM in both float32 and int8 representations.

Model-size table for the requested architectures:

| Width | Parameters | Float32 bytes | Int8 bytes | Float32 300 KiB | Int8 300 KiB | 8 MB PSRAM | MACs |
|---:|---:|---:|---:|---|---|---|---:|
| 32 | 1281 | 5124 | 1281 | Yes | Yes | Yes | 1216 |
| 128 | 17409 | 69636 | 17409 | Yes | Yes | Yes | 17152 |
| 256 | 67585 | 270340 | 67585 | Yes | Yes | Yes | 67072 |
| 512 | 266241 | 1064964 | 266241 | No | Yes | Yes | 265216 |
| 2048 | 4210689 | 16842756 | 4210689 | No | No | No | 4206592 |

For `5 -> w -> w -> 1`, parameters are `w^2 + 7w + 1`.
At float32, width 273 is the last width that fits and width 274 is the first that exceeds
300 KiB. At int8, width 550 is the last width that fits and width 551 is the first that
exceeds 300 KiB.

On an ESP32-S3, assuming one MAC is completed in approximately 1 microsecond on a
reasonably optimized board implementation, the final model's 1216 MACs suggest an
approximate 1.2 ms prediction. This is an engineering estimate, not a board measurement;
the script and model loading would add time.

Calling the server is best for a simple, frequent battery question: the server makes
latency-dependent product behavior and privacy policy easier to change and keeps sensor
data in transit, but network failure removes prediction availability and incurs Wi-Fi
power/network cost. Running the 1281-parameter model locally fits comfortably in flash
and memory, works without Wi-Fi, has lower steady-state power and network latency, and
keeps readings private on the device. A hybrid design should run the model locally and
use the server only for optional updates, diagnostics, or retraining.
