"""Execute the feature-engineering notebook with this environment's fresh kernel."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/feature_engineering.ipynb"
    notebook = nbformat.read(path, as_version=4)
    with TemporaryDirectory(prefix="feature-kernel-") as temporary:
        kernel = Path(temporary) / "kernels" / "feature-engineering"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (feature engineering)", "language": "python",
        }), encoding="utf-8")
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temporary + (os.pathsep + previous if previous else "")
        try:
            NotebookClient(notebook, timeout=1800, kernel_name="feature-engineering",
                           resources={"metadata": {"path": str(root)}}).execute()
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Executed and validated:", path.name)


if __name__ == "__main__":
    main()
