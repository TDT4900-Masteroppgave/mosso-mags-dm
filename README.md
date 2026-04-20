# Hybrid MoSSo

Incremental lossless graph summarization algorithm developed as part of a Master's thesis at NTNU.

This project fuses the exact-cost evaluation engine of **MoSSo (KDD '20)** with the optimization principles of **Mags-DM**, enabling dynamic graph stream processing (vertex/edge insertions and deletions) with competitive compression performance.

---

## Requirements

- Java 11+
- Python 3.8+

---

## Setup

Compile the Java source:

```bash
./compile.sh
```

`run.sh` handles the Python virtual environment and dependencies automatically on first run.

---

## Usage

### Compare algorithms on a local graph file

```bash
./run.sh --type compare --file example_graph.txt
```

### Compare algorithms on remote datasets

```bash
./run.sh --type compare
```

### Incremental vs. batch comparison

```bash
./run.sh --type ivb
```

### Parameter optimization

```bash
# Bayesian optimization
./run.sh --type bayesian --methods mags

# Latin Hypercube Sampling
./run.sh --type lhs --methods mags

# Grid sweep
./run.sh --type sweep --methods mags
```

### Dataset metadata

```bash
./run.sh --type metadata
```

---

## Project Structure

```
.
├── src/                  # Java source (MoSSo algorithm variants)
├── scripts/
│   ├── benchmarks/
│   │   ├── compare.py              # Head-to-head algorithm comparison
│   │   ├── incremental_vs_batch.py # Incremental vs. batch fairness test
│   │   ├── parameter_sweep.py      # Grid search
│   │   ├── latin_hypercube.py      # LHS parameter sampling
│   │   └── bayesian_opt.py         # Bayesian hyperparameter optimization
│   └── dataset_metadata.py
├── datasets/             # Graph datasets
├── output/               # Generated plots and results
├── compile.sh            # Java build script
├── run.sh                # Unified benchmark runner
└── requirements.txt
```

---

## References

- **MoSSo:** Shin et al., *Graph Summarization with Latent Variable Probabilistic Models*, KDD 2020.
- **Mags-DM:** Batch-based graph summarization with divide-and-merge strategies.
