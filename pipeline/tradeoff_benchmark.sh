#!/bin/bash
./run.sh benchmark --group mosso --dynamic DB YT SK LJ \
  --algorithm mosso_fastest_time full_solution_fastest_time \
  mosso_best_compression full_solution_best_compression \
  mosso_balanced full_solution_balanced \
  mosso_default full_solution_bsc full_solution_2_percent_loss
