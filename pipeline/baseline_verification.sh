#!/bin/bash
./run.sh speed --runs 3 --group insertion dynamic --algorithm kdd20-mosso mags &
./run.sh cc --group insertion dynamic  --algorithm kdd20-mosso mags &
./run.sh scale --dataset SK EU --algorithm kdd20-mosso mags &