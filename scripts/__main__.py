"""
Entrypoint for `python -m scripts` (invoked via run.sh).

Usage:
    ./run.sh {compare|sweep|bayesian|ivb|analyze} [options]
"""
import sys
import importlib

_TYPES = {
    "benchmark": "scripts.experiments.benchmark",
    "sweep":     "scripts.experiments.sweep",
    "cot":       "scripts.experiments.compression_over_time",
    "ivb":       "scripts.experiments.incremental_vs_batch",
    "bayesian":  "scripts.experiments.bayesian",
    "analyze":   "scripts.analysis.analyze",
}




def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command not in _TYPES:
        print(f"[!] Unknown command '{command}'.")
        print(__doc__)
        sys.exit(1)

    sys.argv = [sys.argv[0]] + sys.argv[2:]

    module = importlib.import_module(_TYPES[command])
    module.main()


if __name__ == "__main__":
    main()
