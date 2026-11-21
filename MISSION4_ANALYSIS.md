# Mission 4 — Edge AI Feasibility Analysis

## 1. Objective

Determine whether the battery-life prediction neural network can run locally on an ESP32-S3 microcontroller instead of requiring cloud inference.

## 2. Hardware assumptions

- Microcontroller: ESP32-S3
- Available SRAM budget: approximately 300 KiB
- PSRAM: 8 MiB
- Flash: 16 MiB

## 3. Model memory analysis

The network uses five input features and one output.

For an architecture of 5 → w → w → 1:

**Total parameters:**

P = w² + 8w + 1

**Float32 weight memory:**

M_float32 = P × 4 bytes

**Int8 weight memory:**

M_int8 = P × 1 byte

For the default architecture 5 → 32 → 32 → 1:

- Parameters: 1,281
- Float32 weight memory: 5,124 bytes
- Int8 weight memory: 1,281 bytes
- MACs: 1,216
- Largest individual float32 layer output: 128 bytes
- Maximum adjacent input/output buffer estimate: 256 bytes

## 4. Memory feasibility

The default neural network's weights fit within the stated 300 KiB SRAM budget.

Larger architectures require more memory. For the requested configurations, width 256 is the largest tested architecture whose float32 weights fit inside the SRAM budget.

At width 512, the float32 weights exceed the SRAM budget, but the theoretical int8 weight storage fits.

However, weight-storage estimates alone do not guarantee deployment feasibility.

Additional memory is required for activations, temporary buffers, stack, inference runtime and other program resources.

## 5. Cloud versus edge inference

**Cloud advantages:**

- Easier model deployment and updates.
- Greater computational resources.
- Ability to run larger models.

**Cloud disadvantages:**

- Depends on network connectivity.
- Adds communication latency.
- Requires transmitting sensor data.
- May create additional server costs.

**Edge advantages:**

- Can make predictions without internet connectivity.
- Reduces the need to transmit sensor readings.
- Avoids cloud request latency.
- Allows local data processing.

**Edge disadvantages:**

- Limited memory and processing capabilities.
- More complex embedded deployment.
- May require quantisation or optimisation.
- Local inference consumes device energy.

## 6. Conclusion

The default neural network appears to be a promising candidate for edge inference on the ESP32-S3 because it has only 1,281 parameters and requires approximately 5 KiB for float32 weights.

However, memory calculations alone cannot confirm that the model will run successfully on the actual device.

Hardware testing is required to verify execution speed, runtime memory usage, numerical accuracy and power consumption.

A lightweight edge model is worth considering for this doorbell battery-life prediction task.
