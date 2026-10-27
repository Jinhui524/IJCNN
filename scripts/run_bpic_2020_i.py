"""Run Hawkes-DyG on BPIC 2020 I."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hawkes_dyg.runner import main
if __name__ == "__main__":
    main("BPIC_2020_I")
