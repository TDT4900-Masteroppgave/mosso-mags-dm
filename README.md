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

## Overview of Strategies

| Name | Branch | Origin | Parameters | Description | Implemented |
|---|---|---|---|---|---|
| `kdd20-mosso` | main | mosso | e, c, interval | Original MoSSo | :white_check_mark: |
| `cp_mosso_simple` | mosso_strat/cp_mosso_simple | mosso | e, c, interval | Uses MoSSo Simple's candidate pool i.e. candidate pool as neighbors to u | :white_check_mark: |
| `tp_mosso_simple` | mosso_strat/tp_mosso_simple | mosso | e, c, interval | Uses MoSSo Simple's testing pool i.e. a sample of c nodes for the neighbors to u | :x: |
| `sm` | mags_strat/similarity_measure | mags | e, c, interval, h | Uses multiple MinHash to find the most similar candidate to the testing node. The testing node is proposed moved into this candidate instead of a random node from the candidate pool, as in MoSSo | :white_check_mark: |
| `sm_top_b` | mags_strat/similarity_measure_top_b | mags | e, c, interval, h, b | Uses multiple MinHash to find the b most similar candidate to the testing node. The exact delta cost is computed for all these candidates and testing node is proposed moved into the candidate that gives the greatest improvement in representation cost | :white_check_mark: |
| `sm_thr` | mags_strat/similarity_measure_thr | mags | e, c, interval, h, thr_end | Uses multiple MinHash to find the most similar candidate to the testing node. The testing node is proposed moved into this candidate. The threshold is used to only accept candidates with a similarity that exceeds this | :white_check_mark: |
| `ds` | mags_strat/divide_strategy | mags | e, c, interval | Iterates over the coarse clusters instead of the testing nodes | :white_check_mark: |
| `ds_thr` | mags_strat/divide_strategy_thr | mags | e, c, interval, thr_start, thr_end, T | Uses iterations and a decreasing threshold that decreases for each iteration. Each iteration iterates over the coarse clusters. The candidate is selected at random, thereafter are the similarity value between this random candidate and the testing node computed. The move is proposed if the similarity exceeds the current threshold | :white_check_mark: |
| `ds_sm_thr` | mags_strat/divide_strategy_similarity_measure_thr | mags | e, c, interval, h, thr_start, thr_end, T | Uses iterations and a decreasing threshold that decreases for each iteration. Each iteration iterates over the coarse clusters. The most similar candidate is selected and its similarity value are check agians the current threshold. The move is proposed if the similarity exceeds the current threshold | :white_check_mark: |
| `cap` | mags_strat/cap | mags | e, c, interval, cap | Includes a maximum size to the coarse clusters. If the coarse cluster is greater than a defined maximum size, then it is splitted into smaller clusters | :white_check_mark: |
| `sm_top_b_thr` | mags_strat/similarity_measure_top_b_thr | mags | e, c, interval | Uses multiple MinHash to find the b most similar candidate to the testing node. The exact delta cost is computed for all these candidates and testing node is proposed moved into the candidate that gives the greatest improvement in representation cost if the delta cost exceeds the threshold| :x: |
| `saving` | mags_strat/saving_func | mags | e, c, interval | Uses normalized cost to evaluate moves instead of absolute cost | :x: |
| `fix_testing_nodes` | strat/fix_testing_nodes | authors | e, c, interval | Ensures that only unique nodes exists in the testing nodes | :white_check_mark: |
| `excl_coarse_clustering` | strat/excl_coarse_clustering | authors | e, c, interval | Original MoSSo, but does not make use of coarse clusters | :x: |
| `sm_2hop` | strat/similarity_measure_2hop | authors | e, c, interval | Uses multiple MinHash to find the most similar candidate to the testing node. The testing node is proposed moved into this candidate instead of a random node from the candidate pool. Also uses a sample for the 2hop neighborhood | :white_check_mark: |
| `tp_similar_nbrs` | strat/tp_similar_nbrs | authors | e, c, interval | Constructs the testing pool by selecting the top-k most similar neighbors of the processing node u. Any remaining capacity is filled by duplicating nodes in descending order of similarity | :x: |
| `random_top_b` | strat/random_top_b | authors | e, c, interval | Selects b nodes at random from the candidate pool. Computes the delta cost to each of these and selects the candidate that gives the greatest improvement in representation cost | :white_check_mark: |
| `prob_selection_similarity` | strat/prob_selection_similarity | authors | e, c, interval | Selects candidate nodes with probabilities proportional to their similarity i.e. sim(z)/total_sim | :white_check_mark: |
| `one_processing_src_dst` | strat_deprecated/one_processing_src_dst | authors | e, c, interval |Deprecated. The idea was to process both the src and the dst i the same trial as their affected nodes most likely are very similar | :white_check_mark: |
| `sm_opt` | strat_deprecated/similarity_measure_optimalization | authors | e, c, interval | Deprecated | :white_check_mark: |

### Parameters

| Name | Description |
|---|---|
| `c` | Sample number | 
| `e` | Probability for corrective escape | 
| `interval` | Interval | 
| `b` | Number of top-b candidates | 
| `h` | Number of hash functions |
| `thr_start` | Start value for a dynamic decreasing threshold | 
| `thr_end` | End value for a dynamic decreasing threshold | 
| `cap` | Maximum size for coarse clusters | 
| `T` | Number of iterations | 



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
