# Hybrid MoSSo: Lossless Graph Summarization (MoSSo + Mags-DM)

**Hybrid MoSSo** is an incremental algorithm for lossless graph summarization, developed as part of a Master's thesis at
NTNU.

This project represents a complete fusion of two state-of-the-art approaches: it leverages the dynamic, exact-cost
evaluation engine of the original **MoSSo (KDD '20)** and upgrades it by implementing the full suite of **Mags-DM**
optimization principles.

### Key Features

* 🚀 **Full Mags-DM Integration:** A complete implementation of Mags-DM principles inside an incremental environment.
* ⚡ **Dynamic & Incremental:** Processes graph streams (insertions/deletions) in near-constant time.
* 📊 **Automated Benchmarking Suite:** A fully parameterized Python pipeline to seamlessly download
  datasets, compile Java code, and generate comparative performance plots against the original KDD '20 baseline.

---

## Benchmarking

To run the benchmark comparing the original MoSSo against this Hybrid implementation:

### Run on a local graph file
```bash
python3 benchmark/compare.py --mode local --file example_graph.txt
```

### Run the full remote dataset suite
```bash
python3 benchmark/compare.py --mode remote
```

