#!/bin/bash

# Array of sample numbers for the sweep
SAMPLE_SIZES=(240 120 60 30)

for c_val in "${SAMPLE_SIZES[@]}"; do
    echo "Starting parameter sweeps for sample size (c) = $c_val"

    ./run.sh sweep --algorithm sm --dataset PR EN FB DB YT --dynamic DB YT --param h --range 5 50 5 --c "$c_val" &
    sleep 2

    ./run.sh sweep --algorithm sm --dataset PR EN FB DB YT --dynamic DB YT --param h --range 1 1 1 --c "$c_val" &
    sleep 2

    ./run.sh sweep --algorithm sm_thr --dataset PR EN FB DB YT --dynamic DB YT --param thr --range 0.0 1.1 0.1 --c "$c_val" &
    sleep 2

    ./run.sh sweep --algorithm top-b --dataset PR EN FB DB YT --dynamic DB YT --param b --range 1 10 1 --c "$c_val" &
    sleep 2

    ./run.sh sweep --algorithm cap --dataset PR EN FB DB YT --dynamic DB YT --param cap --range 0 120 10 --c "$c_val" &
    sleep 2

    ./run.sh sweep --algorithm ds_thr --dataset PR EN FB DB YT --dynamic DB YT --param T --range 10 100 10 --c "$c_val" &
    sleep 2
    ./run.sh sweep --algorithm ds_thr --dataset PR EN FB DB YT --dynamic DB YT --param T --range 1 1 1 --c "$c_val" &

    # Wait for all background sweeps of the current sample size to finish before moving to the next
    wait
    echo "Completed parameter sweeps for sample size (c) = $c_val"
done