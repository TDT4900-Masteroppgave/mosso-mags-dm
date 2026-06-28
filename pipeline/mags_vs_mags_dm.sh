#!/bin/bash
./run.sh benchmark --runs 3 --group mags_small mags_large --algorithm mags mags_dm para_mags para_mags_dm --p 30

./run.sh speed --runs 3 --dataset PR EN FB YT SK EU LJ --dynamic YT SK LJ --algorithm mags_dm
./run.sh cc --dataset PR EN FB YT SK EU LJ --dynamic DB YT SK LJ --algorithm mags_dm