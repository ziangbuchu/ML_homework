"""Print dependency and hardware information for the ML homework environment."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "tqdm",
]


def _version(module_name: str) -> str:
    package_names = {
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
    }
    package_name = package_names.get(module_name, module_name)
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    _log(f"python={sys.version.split()[0]}")
    _log(f"platform={platform.platform()}")

    for module_name in PACKAGES:
        _log(f"{module_name}={_version(module_name)}")

    import torch

    _log(f"torch_cuda_available={torch.cuda.is_available()}")
    _log(f"torch_cuda_version={torch.version.cuda}")
    _log(f"torch_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            _log(f"torch_device_{index}={torch.cuda.get_device_name(index)}")


if __name__ == "__main__":
    main()
