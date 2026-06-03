"""
Jupyter kernel startup script.
Loaded automatically via jupyter.runStartupCommands in .vscode/settings.json
every time a new kernel starts.

First-time environment setup (run once from the project root):
    pip install -r requirements-dev.txt

This installs the project in editable mode (pyproject.toml is the source of
truth for runtime deps), so no sys.path manipulation is needed.
The fallback below handles the case where the package has not been installed yet.
"""
import os
import sys

# ── sys.path fallback (remove once `pip install -e .` has been run) ───────────
try:
    import pipeline  # noqa: F401 — succeeds if installed via pyproject.toml
except ModuleNotFoundError:
    _HERE = os.path.dirname(os.path.abspath(__file__))   # notebooks/
    _ROOT = os.path.dirname(_HERE)                        # project root
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# ── pipeline imports ──────────────────────────────────────────────────────────
import importlib

import pipeline.config as config
import pipeline.zoi as zoi
import pipeline.perception as perception
import pipeline.safety_logic as safety_logic
import pipeline.visualization as visualization
import pipeline.runner as runner

for _mod in [config, zoi, perception, safety_logic, visualization, runner]:
    importlib.reload(_mod)

from pipeline.runner import run_pipeline, PipelineResult

import cv2
import numpy as np
import matplotlib.pyplot as plt
_ip = get_ipython()  # noqa: F821
if _ip is not None:
    _ip.run_line_magic("matplotlib", "inline")

print(f"OpenCV : {cv2.__version__}")
print(f"NumPy  : {np.__version__}")
print("Pipeline package loaded.")
