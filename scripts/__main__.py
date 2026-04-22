"""
Entrypoint for `python -m scripts` (invoked via run.sh).

Usage:
    ./run.sh --type {compare|sweep|lhs|bayesian|ivb|significance} [options]
"""
import sys


_TYPES = {
    "compare":     "scripts.benchmarks.compare",
    "sweep":       "scripts.benchmarks.parameter_sweep",
    "lhs":         "scripts.benchmarks.latin_hypercube",
    "bayesian":    "scripts.benchmarks.bayesian_opt",
    "ivb":         "scripts.benchmarks.incremental_vs_batch",
    "significance":"scripts.benchmarks.significance_test",
    "analyze":     "scripts.analyze",
}


def _parse_type(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--type" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--type="):
            return arg.split("=", 1)[1]
    return None


def _strip_type(argv: list[str]) -> list[str]:
    """Return argv with --type / --type=VALUE removed so benchmark parsers don't see it."""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--type":
            i += 2  # skip flag and its value
        elif argv[i].startswith("--type="):
            i += 1  # skip combined flag=value
        else:
            out.append(argv[i])
            i += 1
    return out


def main() -> None:
    bench_type = _parse_type(sys.argv[1:])

    if bench_type not in _TYPES:
        types_str = "|".join(_TYPES)
        print(f"Usage: ./run.sh --type {{{types_str}}} [options]")
        print("Example: ./run.sh --type compare --algorithm mags --group small")
        sys.exit(1)

    # Remove --type from argv so the benchmark's own argparse doesn't see it
    sys.argv = [sys.argv[0]] + _strip_type(sys.argv[1:])

    import importlib
    module = importlib.import_module(_TYPES[bench_type])
    # Each module has an `if __name__ == "__main__"` guard that instantiates and runs its benchmark.
    # We replicate that by calling the benchmark class directly.
    if bench_type == "analyze":
        module.main()
        return

    _CLASS = {
        "scripts.benchmarks.compare":              "CompareBenchmark",
        "scripts.benchmarks.parameter_sweep":      "ParameterSweepBenchmark",
        "scripts.benchmarks.latin_hypercube":      "LHSBenchmark",
        "scripts.benchmarks.bayesian_opt":         "BayesianOptimizationBenchmark",
        "scripts.benchmarks.incremental_vs_batch": "IncrementalVsBatchBenchmark",
        "scripts.benchmarks.significance_test":    "SignificanceTestBenchmark",
    }

    cls = getattr(module, _CLASS[_TYPES[bench_type]])
    cls().run()


if __name__ == "__main__":
    main()
