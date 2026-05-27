#!/bin/bash
./run.sh speed --runs 3 --group insertion dynamic --dynamic DB YT SK LJ --algorithm MoSSo full_solution mags
./run.sh cc --group insertion dynamic --dynamic DB YT SK LJ --algorithm MoSSo full_solution mags
./run.sh scale --dataset SK EU --algorithm MoSSo full_solution mags
./run.sh benchmark --runs 3 --group insertion dynamic --dynamic DB YT SK LJ --algorithm MoSSo full_solution