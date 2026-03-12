#!/usr/bin/env python3
from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }

    if payload["cuda_available"]:
        payload["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
