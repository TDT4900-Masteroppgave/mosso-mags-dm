#!/bin/bash
./run.sh speed --runs 3 --group insertion dynamic --dynamic DB YT SK LJ --algorithm kdd20-mosso proposed kags
./run.sh cc --group insertion dynamic --dynamic DB YT SK LJ --algorithm kdd20-mosso proposed mags
./run.sh scale --dataset SK EU --algorithm kdd20-mosso proposed mags
./run.sh benchmark --runs 3 --group insertion dynamic --dynamic DB YT SK LJ --algorithm kdd20-mosso proposed