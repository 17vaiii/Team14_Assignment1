"""model.py -- the network, shared by train.py and the server so both build the same thing."""
import torch
from torch import nn

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]


class BatteryNet(nn.Module):
    def __init__(self, hidden=(32, 32)):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def model_size_info(model):
    layers = []
    total_params = 0
    total_bytes_f32 = 0
    total_macs = 0

    for module in model.net:
        if isinstance(module, nn.Linear):
            params = sum(parameter.numel() for parameter in module.parameters())
            bytes_f32 = params * 4
            macs = module.in_features * module.out_features
            layers.append({
                "name": f"linear{len(layers) + 1}",
                "in": module.in_features,
                "out": module.out_features,
                "params": params,
                "bytes_f32": bytes_f32,
                "macs": macs,
            })
            total_params += params
            total_bytes_f32 += bytes_f32
            total_macs += macs

    total_bytes_int8 = total_params
    sram_limit_bytes = 300 * 1024
    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": total_macs,
        "fits_300kb_sram_f32": total_bytes_f32 <= sram_limit_bytes,
        "fits_300kb_sram_int8": total_bytes_int8 <= sram_limit_bytes,
    }
