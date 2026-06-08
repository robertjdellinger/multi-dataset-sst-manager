#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  scripts/global_sst_reconstruction_reference_plot.py
#
#  Compatibility wrapper for the canonical SST reference figure and annual
#  diagnostics script. Keep this path for older commands, but maintain the
#  implementation in scripts/sea_surface_temperature/plot_global_sst_reference_figure.py.

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sea_surface_temperature.plot_global_sst_reference_figure import main  # noqa: E402


if __name__ == "__main__":
    main()
