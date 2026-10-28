"""Compatibility entry point for the public Unitree G1 VLA evaluator.

Use ``python -m robocasa_eval`` for a dependency-light planning command.
This file also supports ``python robocasa/scripts/evaluate.py ...`` without
importing RoboCasa and its simulation dependencies before CLI dispatch.
"""

from pathlib import Path
import sys


def main():
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robocasa_eval.cli import main as evaluate

    return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
