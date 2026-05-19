import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluation.base_evaluate import main


if __name__ == "__main__":
    main("lean")
