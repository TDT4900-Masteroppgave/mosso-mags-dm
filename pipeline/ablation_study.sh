#!/bin/bash
./run.sh benchmark \
    --runs 3 \
    --group insertion dynamic --dynamic DB YT SK LJ  \
    --algorithm kdd20-mosso sm full_solution \
    --c 50 \
    --e 2 \
    --h 5