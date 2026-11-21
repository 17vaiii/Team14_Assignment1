
# Mission 4 - Edge AI Memory Analysis

SRAM_LIMIT = 300 * 1024        # 300 KiB
PSRAM_LIMIT = 8 * 1024 * 1024  # 8 MiB


def analyse_model(hidden_width):
    # Network: 5 -> hidden -> hidden -> 1
    layers = [
        (5, hidden_width),
        (hidden_width, hidden_width),
        (hidden_width, 1)
    ]

    total_params = 0
    total_macs = 0

    for inputs, outputs in layers:
        parameters = (inputs * outputs) + outputs
        macs = inputs * outputs

        total_params += parameters
        total_macs += macs

    float32_bytes = total_params * 4
    int8_bytes = total_params

    # Peak single-layer output activation
    activation_bytes = hidden_width * 4

    return {
        "width": hidden_width,
        "params": total_params,
        "float32_bytes": float32_bytes,
        "int8_bytes": int8_bytes,
        "activation_bytes": activation_bytes,
        "macs": total_macs,
        "fits_sram_float32": float32_bytes <= SRAM_LIMIT,
        "fits_sram_int8": int8_bytes <= SRAM_LIMIT,
        "fits_psram_float32": float32_bytes <= PSRAM_LIMIT
    }


widths = [32, 128, 256, 512, 2048]

print("MISSION 4 - EDGE AI ANALYSIS")
print("-" * 85)

for width in widths:
    result = analyse_model(width)

    print(f"\nArchitecture: 5 -> {width} -> {width} -> 1")
    print(f"Parameters: {result['params']:,}")
    print(f"Float32 weight bytes: {result['float32_bytes']:,}")
    print(f"Int8 weight bytes: {result['int8_bytes']:,}")
    print(f"Activation bytes (single output): {result['activation_bytes']:,}")
    print(f"MACs: {result['macs']:,}")
    print(f"Float32 weights fit SRAM: {result['fits_sram_float32']}")
    print(f"Int8 weights fit SRAM: {result['fits_sram_int8']}")
    print(f"Float32 weights fit PSRAM: {result['fits_psram_float32']}")
