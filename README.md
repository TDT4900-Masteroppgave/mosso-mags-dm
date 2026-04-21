# MoSSo vs. MAGS Graph Summarization Benchmark

A comprehensive, automated benchmarking framework designed to evaluate, compare, and optimize incremental (streaming) graph summarization algorithms (MoSSo) against batch graph summarization algorithms (MAGS).

This repository was built to test custom merging strategies and optimizations developed for MoSSo, tracking metrics like execution time, compression ratio, and state-update costs over time.

## Features

**Automated Code Management:** Dynamically clones specific branches from remote Git repositories, applies hotfixes, and compiles Java (`.jar`) and C++ (CMake) binaries on the fly.

**Dataset Orchestration:** Automatically downloads, unzips, and formats real-world graph datasets (from SNAP) like Enron, Caida, DBLP, and LiveJournal.

**Intelligent Pre-processing:** Cleans raw datasets by removing self-loops, standardizing edge directionality, and removing duplicate edges to align with KDD '20 methodology.

**Rich Visualization:** Automatically generates comprehensive outputs for every run, including formatted CSVs, terminal tables, and high-quality PDF plots.

## Prerequisites & Installation

- Python 3.9+
- Java 12+ (required for compiling and running MoSSo)
- CMake & Clang/GCC (required for compiling MAGS C++ binaries)
- Git

```bash
# Clone the repository
git clone https://github.com/TDT4900-Masteroppgave/mosso-mags-dm.git
cd mosso-mags-dm

# Install required Python packages
pip install -r requirements.txt
```

## Benchmark Modes

The framework is driven by the `./run.sh` script, which routes commands to various Python benchmark modules. Use the `--type` flag to select your benchmark mode.

### 1. `compare` — Standard Comparison

Runs the specified algorithms on full datasets to measure overall execution time and compression ratio. If `--runs` is set to >1, it calculates standard deviation and plots variance.

```bash
./run.sh --type compare --algorithm kdd20-mosso mags strat_1 --dataset CA --runs 3
```

### 2. `ivb` — Streaming vs. Checkpoint Evolution

Replicates the methodology from the MoSSo paper. Evaluates how compression quality and update costs evolve over time as the graph grows (e.g., at 20%, 40%, 60%, 80%, 100% of edges).

- **Incremental:** Processes the stream once and calculates the per-edge update cost.
- **Batch:** Re-runs from scratch at every checkpoint to demonstrate the growing cost of batch summarization.

```bash
./run.sh --type ivb --algorithm kdd20-mosso mags --dataset EN
```

### 3. `bayesian` — Hyperparameter Optimization

Uses Optuna to perform Bayesian Optimization across algorithm parameters (e.g., sample size, escape probability, thresholds). Automatically finds and plots the Pareto front (tradeoff between time and compression ratio).

```bash
./run.sh --type bayesian --algorithm strat_1 strat_2_thr --dataset CA --iterations 50 --jobs -1
```

### 4. `sweep` / `lhs` — Parameter Sweeping

- **`sweep`:** Sweeps through a linear range of a single parameter to graph its direct impact on performance.
- **`lhs`:** Uses Latin Hypercube Sampling to efficiently sample multidimensional parameter spaces.

```bash
./run.sh --type sweep --algorithm kdd20-mosso strat_1 --param c --range 10 240 20 --dataset CA
./run.sh --type lhs --algorithm strat_1 strat_2 --dataset CA --samples 30
```

## Datasets

| ID | Name | Nodes | Edges |
|---|---|---|---|
| `CA` | AS-CAIDA | 26,475 | 53,381 |
| `EN` | Email-Enron | 36,692 | 183,831 |
| `BK` | Brightkite | 58,228 | 214,078 |
| `EA` | Email-EuAll | 265,009 | 364,481 |
| `SL` | Slashdot | 82,168 | 504,230 |
| `DB` | DBLP | 317,080 | 1,049,866 |
| `AM` | Amazon | 403,394 | 2,443,408 |
| `YT` | YouTube | 1,134,890 | 2,987,624 |
| `SK` | AS-Skitter | 1,696,415 | 11,095,298 |
| `LJ` | LiveJournal | 3,997,962 | 34,681,189 |

Use `--dataset CA EN` to select specific datasets, `--group small` for the first six, or `--group large` for the last four.

## Output & Artifacts

Every benchmark run creates a timestamped folder at `output/benchmarks/<type>/run_YYYYMMDD_HHMMSS/` containing:

| File | Description |
|---|---|
| `execution.log` | Full terminal output and internal logging |
| `results.csv` / `results_raw.csv` | Raw metrics from the run |
| `table_results.txt` | Formatted text table of results |
| `*.pdf` | Matplotlib plots (Pareto front, boxplots, parameter analysis, etc.) |
| `summarized_graphs/` | Raw output files from the Java/C++ binaries |

## References

- **MoSSo:** Ko, J. et al. *Incremental Lossless Graph Summarization.* KDD 2020.
- **MAGS:** Chu, D. et al. *Graph Summarization: Compactness Meets Efficiency.* VLDB 2024.
