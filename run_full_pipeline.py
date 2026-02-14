"""Repository-root wrapper for the simulation study pipeline.

Allows running:
    python run_full_pipeline.py ...
from the repository root on Windows/Linux/macOS.
"""

from pathlib import Path
import os
import sys
import runpy


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sim_dir = repo_root / "Paper_abgabe" / "simulation_study"
    target_script = sim_dir / "run_full_pipeline.py"

    if not target_script.exists():
        raise FileNotFoundError(f"Expected script not found: {target_script}")

    # Keep behavior identical to running inside Paper_abgabe/simulation_study
    # and ensure local imports (e.g. `from utils import ...`) resolve on all platforms.
    os.chdir(sim_dir)
    sim_dir_str = str(sim_dir)
    if sim_dir_str not in sys.path:
        sys.path.insert(0, sim_dir_str)

    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
