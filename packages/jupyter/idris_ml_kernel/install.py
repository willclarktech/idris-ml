"""Install the idris-ml Jupyter kernel spec."""

import json
import os
import sys
import tempfile
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager


def main():
    """Install kernel.json into Jupyter's kernel directory."""
    # Find project root (packages/jupyter/idris_ml_kernel/ → 4 levels up)
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    kernel_spec = {
        "argv": [sys.executable, "-m", "idris_ml_kernel", "-f", "{connection_file}"],
        "display_name": "Idris 2 (idris-ml)",
        "language": "idris2",
        "env": {"IDRIS_ML_ROOT": str(project_root)},
    }

    with tempfile.TemporaryDirectory() as td:
        spec_dir = Path(td) / "idris-ml"
        spec_dir.mkdir()
        with open(spec_dir / "kernel.json", "w") as f:
            json.dump(kernel_spec, f, indent=2)

        ksm = KernelSpecManager()
        dest = ksm.install_kernel_spec(
            str(spec_dir), kernel_name="idris-ml", user=True
        )
        print(f"Installed kernel spec to {dest}")


if __name__ == "__main__":
    main()
