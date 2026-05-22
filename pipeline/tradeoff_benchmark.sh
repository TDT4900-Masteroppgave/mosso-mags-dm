#!/bin/bash
./run.sh benchmark --runs 3 --group insertion dynamic --dynamic DB YT SK LJ \
  --algorithm mosso_default mosso_best_compression mosso_fastest_time mosso_balanced \
  full_solution_best_compression full_solution_fastest_time full_solution_balanced \
  full_solution_bsc full_solution_bst