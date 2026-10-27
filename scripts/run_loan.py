"""Run the first Hawkes-DyG experiment on the loan application log."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawkes_dyg.runner import main  # noqa: E402


if __name__ == "__main__":
    main("Loan_Application_events")
