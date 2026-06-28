#!/bin/bash
./run.sh benchmark --group mosso mags --dynamic DB YT SK LJ --algorithm MoSSo

./run.sh benchmark --runs 3 --group mosso mags --algorithm mags mags_dm para_mags para_mags_dm --p 32

./run.sh cc --group mosso --dynamic DB YT SK LJ --algorithm MoSSo full_solution mags

./run.sh speed --runs 3 --dataset PR EN FB EU SK IC UK-05 IT --dynamic SK --algorithm mags_dm
./run.sh speed --runs 3 --group mosso --dynamic DB YT SK LJ --algorithm MoSSo full_solution mags

./run.sh scale --dataset SK EU LJ HW UK --dynamic SK LJ --algorithm MoSSo full_solution

./run.sh benchmark --runs 3 --group mosso --dynamic DB YT SK LJ \
  --algorithm mosso_fastest_time full_solution_fastest_time \
  mosso_best_compression full_solution_best_compression \
  mosso_balanced full_solution_balanced \
  mosso_default full_solution_bsc full_solution_2_percent_loss

./run.sh benchmark --runs 3 --group mosso --dynamic DB YT SK LJ --algorithm MoSSo full_solution full_solution_sm
