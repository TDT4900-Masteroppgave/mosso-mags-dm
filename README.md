# MoSSo vs. MAGS Graph Summarization Benchmark

A benchmarking framework designed to evaluate, compare, and optimize incremental (streaming) graph summarization algorithms (MoSSo) against batch graph summarization algorithms (MAGS).

This repository was built to test custom merging strategies and optimizations developed for MoSSo, tracking metrics like execution time and compression ratio.

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

| Name                        | Branch                                             | Origin  | Parameters                               | Implemented        |
|-----------------------------|----------------------------------------------------|---------|------------------------------------------|--------------------|
| `kdd20-mosso`               | main                                               | mosso   | e, c, interval                           | :white_check_mark: |
| `cp_mosso_simple`           | mosso_strat/cp_mosso_simple                        | mosso   | e, c, interval                           | :white_check_mark: |
| `tp_mosso_simple`           | mosso_strat/tp_mosso_simple                        | mosso   | e, c, interval                           | :x:                |
| `sm`                        | mags_strat/similarity_measure                      | mags    | e, c, interval, h                        | :white_check_mark: |
| `sm_top_b`                  | mags_strat/similarity_measure_top_b                | mags    | e, c, interval, h, b                     | :white_check_mark: |
| `sm_thr`                    | mags_strat/similarity_measure_thr                  | mags    | e, c, interval, h, thr_end               | :white_check_mark: |
| `ds`                        | mags_strat/divide_strategy                         | mags    | e, c, interval                           | :white_check_mark: |
| `ds_thr`                    | mags_strat/divide_strategy_thr                     | mags    | e, c, interval, thr_start, thr_end, T    | :white_check_mark: |
| `ds_sm_thr`                 | mags_strat/divide_strategy_similarity_measure_thr  | mags    | e, c, interval, h, thr_start, thr_end, T | :white_check_mark: |
| `cap`                       | mags_strat/cap                                     | mags    | e, c, interval, cap                      | :white_check_mark: |
| `sm_top_b_thr`              | mags_strat/similarity_measure_top_b_thr            | mags    | e, c, interval                           | :x:                |
| `saving`                    | mags_strat/saving_func                             | mags    | e, c, interval                           | :x:                |
| `fix_testing_nodes`         | strat/fix_testing_nodes                            | authors | e, c, interval                           | :white_check_mark: |
| `excl_coarse_clustering`    | strat/excl_coarse_clustering                       | authors | e, c, interval                           | :x:                |
| `sm_2hop`                   | strat/similarity_measure_2hop                      | authors | e, c, interval                           | :white_check_mark: |
| `tp_similar_nbrs`           | strat/tp_similar_nbrs                              | authors | e, c, interval                           | :x:                |
| `random_top_b`              | strat/random_top_b                                 | authors | e, c, interval                           | :white_check_mark: |
| `prob_selection_similarity` | strat/prob_selection_similarity                    | authors | e, c, interval                           | :white_check_mark: |
| `one_processing_src_dst`    | strat_deprecated/one_processing_src_dst            | authors | e, c, interval                           | :white_check_mark: |
| `sm_opt`                    | strat_deprecated/similarity_measure_optimalization | authors | e, c, interval                           | :white_check_mark: |

### `kdd20-mosso` 
Original MoSSo
### `cp_mosso_simple`
Uses MoSSo Simple's candidate pool i.e., candidate pool as neighbors to u.
### `tp_mosso_simple` 
Uses MoSSo Simple's testing pool i.e., a sample of c nodes for the neighbors to u.
### `sm` 
Uses multiple MinHash to find the most similar candidate to the testing node. The testing node is proposed moved into 
this candidate instead of a random node from the candidate pool, as in MoSSo.
### `sm_top_b`
Uses multiple MinHash to find the b most similar candidate to the testing node. The exact delta cost is computed for 
all these candidates, and the testing node is proposed moved into the candidate that gives the greatest improvement in 
representation cost.
### `sm_thr`
Uses multiple MinHash to find the most similar candidate to the testing node. The testing node is proposed moved into 
this candidate. The threshold is used to only accept candidates with a similarity that exceeds this.
### `ds`
Iterates over the coarse clusters instead of the testing nodes.
### `ds_thr`
Uses iterations and a decreasing threshold that decreases for each iteration. Each iteration iterates over the coarse 
clusters. The candidate is selected at random, thereafter are the similarity value between this random candidate and the 
testing node computed. The move is proposed if the similarity exceeds the current threshold.
### `ds_sm_thr`
Uses iterations and a decreasing threshold that decreases for each iteration. Each iteration iterates over the coarse 
clusters. The most similar candidate is selected, and its similarity value is checked against the current threshold. The 
move is proposed if the similarity exceeds the current threshold.
### `cap` , cap | Includes a maximum size to the coarse clusters. If the coarse cluster is greater than a defined 
maximum size, then it is split into smaller clusters.
### `sm_top_b_thr`
Uses multiple MinHash to find the b most similar candidate to the testing node. The exact delta cost is computed for all 
these candidates, and the testing node is proposed moved into the candidate that gives the greatest improvement in 
representation cost if the delta cost exceeds the threshold| :x: |
### `saving`
Uses normalized cost to evaluate moves instead of absolute cost.
### `fix_testing_nodes`
Ensures that only a unique node exists in the testing nodes.
### `excl_coarse_clustering`
Original MoSSo, but does not make use of coarse clusters.
### `sm_2hop`
Uses multiple MinHash to find the most similar candidate to the testing node. The testing node is proposed moved into 
this candidate instead of a random node from the candidate pool. Also uses a sample for the 2hop neighborhood.
### `tp_similar_nbrs`
Constructs the testing pool by selecting the top-k most similar neighbors of the processing node u. Any remaining 
capacity is filled by duplicating nodes in descending order of similarity.
### `random_top_b`
Selects b nodes at random from the candidate pool. Computes the delta cost to each of these and selects the candidate 
that gives the greatest improvement in representation cost.
### `prob_selection_similarity`
Selects candidate nodes with probabilities proportional to their similarity, i.e., sim(z)/total_sim.
### `one_processing_src_dst`
Deprecated. The idea was to process both the src and the dst in the same trial as their affected nodes most likely are very similar.
### `sm_opt`
Deprecated.

### Parameters

| Name        | Description                                    |
|-------------|------------------------------------------------|
| `c`         | Sample number                                  | 
| `e`         | Probability for corrective escape              | 
| `interval`  | Interval                                       | 
| `b`         | Number of top-b candidates                     | 
| `h`         | Number of hash functions                       |
| `thr_start` | Start value for a dynamic decreasing threshold | 
| `thr_end`   | End value for a dynamic decreasing threshold   | 
| `cap`       | Maximum size for coarse clusters               | 
| `T`         | Number of iterations                           | 

## Usage

The framework is driven by `./run.sh`, which dispatches to subcommands defined in `scripts/__main__.py`:

```
./run.sh {benchmark|sweep|cot|ivb|bayesian|analyze} [options]
```

### Common Flags

Shared by all experiment subcommands (`benchmark`, `sweep`, `cot`, `ivb`, `bayesian`):

| Flag                                                         | Description                                             |
|--------------------------------------------------------------|---------------------------------------------------------|
| `--runs <n>`                                                 | Repetitions per configuration (default `1`)             |
| `--group {all,small,large}`                                  | Dataset group (default `all`)                           |
| `--dataset <ID ...>`                                         | Specific datasets; overrides `--group`                  |
| `--algorithm <name ...>`                                     | Algorithms to run                                       |
| `--is-local`                                                 | Include the local working tree as the `local` algorithm |
| `--seed <int>`                                               | Random seed (default `42`)                              |
| `--c --e --interval --b --h --thr_start --thr_end --cap --T` | Override parameter defaults                             |

### 1. `benchmark` — Standard Comparison

Runs algorithms on full datasets, measuring execution time and compression ratio. Use `--baseline <algo>` for relative comparisons; with `--runs > 1` standard deviation is reported.

```bash
./run.sh benchmark --algorithm kdd20-mosso mags strat_2 --dataset CA --runs 3 --baseline kdd20-mosso
```

### 2. `ivb` — Streaming vs. Batch

Compares incremental (MoSSo) vs. batch (MAGS) algorithms. MoSSo execution time is normalized to microseconds-per-edge for fair comparison.

```bash
./run.sh ivb --algorithm kdd20-mosso mags --dataset EN
```

### 3. `cot` — Compression Over Time

Evaluates algorithms at edge-stream checkpoints (e.g., 20%, 40%, 60%, 80%, 100%) to track how compression evolves as the graph grows.

| Flag                        | Description                                                 |
|-----------------------------|-------------------------------------------------------------|
| `--checkpoints <f1 f2 ...>` | Stream fractions in `(0,1]` (default `0.2 0.4 0.6 0.8 1.0`) |

```bash
./run.sh cot --algorithm kdd20-mosso strat_2 --dataset YT --checkpoints 0.2 0.5 1.0
```

### 4. `sweep` — Parameter Sweep

Sweeps a single hyperparameter to study its sensitivity. Provide either `--range` or `--values`.

| Flag                                                 | Description                     |
|------------------------------------------------------|---------------------------------|
| `--param {c,e,interval,b,h,thr_start,thr_end,cap,T}` | Parameter to sweep (required)   |
| `--range <start> <stop> <step>`                      | Three ints, passed to `range()` |
| `--values <v1 v2 ...>`                               | Explicit values                 |

```bash
./run.sh sweep --algorithm kdd20-mosso strat_2 --param c --range 10 240 20 --dataset CA
./run.sh sweep --algorithm strat_2 --param h --values 4 6 8 10 12 --runs 2
```

### 5. `bayesian` — Hyperparameter Optimization

Runs Optuna multi-objective optimization (minimize time and compression ratio) and stores results in an Optuna SQLite study at `<session>/optuna_study.db`. The Pareto front is exposed by the `analyze` command.

| Flag           | Description                                      |
|----------------|--------------------------------------------------|
| `--trials <n>` | Optuna trials per algorithm (default `100`)      |
| `--jobs <n>`   | Parallel jobs; `-1` uses all cores (default `1`) |

```bash
./run.sh bayesian --algorithm strat_2 strat_2_thr --dataset CA --trials 100 --jobs -1
```

### 6. `analyze` — Interactive Post-processing

Interactive CLI (no flags). Scans `output/experiments/`, prompts for experiment type, session, algorithms, datasets, aggregation, and plot type, then writes plots/CSVs to `<session>/analysis/`.

Available plot types per experiment:

| Experiment  | Plots                                                        |
|-------------|--------------------------------------------------------------|
| `benchmark` | bar (time & ratio)                                           |
| `ivb`       | log-scale bar (streaming vs. batch)                          |
| `cot`       | line (compression over checkpoints)                          |
| `sweep`     | parameter sensitivity lines                                  |
| `bayesian`  | Pareto front, knee point, marginal utility, reverse engineer |

```bash
./run.sh analyze
```

## Datasets

| ID   | Name        | Nodes     | Edges      |
|------|-------------|-----------|------------|
| `CA` | AS-CAIDA    | 26,475    | 53,381     |
| `EN` | Email-Enron | 36,692    | 183,831    |
| `BK` | Brightkite  | 58,228    | 214,078    |
| `EA` | Email-EuAll | 265,009   | 364,481    |
| `SL` | Slashdot    | 82,168    | 504,230    |
| `DB` | DBLP        | 317,080   | 1,049,866  |
| `AM` | Amazon      | 403,394   | 2,443,408  |
| `YT` | YouTube     | 1,134,890 | 2,987,624  |
| `SK` | AS-Skitter  | 1,696,415 | 11,095,298 |
| `LJ` | LiveJournal | 3,997,962 | 34,681,189 |

Use `--dataset CA EN` to select specific datasets, `--group small` for the first six, or `--group large` for the last four.

## Output & Artifacts

Every run creates a timestamped folder at `output/experiments/<type>/run_YYYYMMDD_HHMMSS/` containing:

| File                              | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `execution.log`                   | Full terminal output and internal logging       |
| `results.csv` / `results_raw.csv` | Raw metrics from the run                        |
| `results.db`                      | SQLite results database (consumed by `analyze`) |
| `table_results.txt`               | Formatted text table of results                 |
| `optuna_study.db`                 | Optuna study (Bayesian runs only)               |
| `analysis/`                       | Plots and CSVs produced by `./run.sh analyze`   |
| `summarized_graphs/`              | Raw output files from the Java/C++ binaries     |

## References

- **MoSSo:** Ko, J. et al. *Incremental Lossless Graph Summarization.* KDD 2020.
- **MAGS:** Chu, D. et al. *Graph Summarization: Compactness Meets Efficiency.* VLDB 2024.
