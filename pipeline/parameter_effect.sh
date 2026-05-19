#!/bin/bash
./run.sh sweep --algorithm sm --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param h --range 5 50 5 &
sleep 2
./run.sh sweep --algorithm sm --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param h --range 1 5 1 &
sleep 2

./run.sh sweep --algorithm sm_thr --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param thr --range 0.1 1.0 0.1 &&
sleep 2

./run.sh sweep --algorithm top-b --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param b --range 1 10 1 &
sleep 2

./run.sh sweep --algorithm cap --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param cap --range 10 120 10 &
sleep 2

( ./run.sh sweep --algorithm ds_thr --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param thr --range 0.1 1.0 0.1 && \
  ./run.sh sweep --algorithm ds_thr --dataset PR EN FB EU DB YT SK LJ --dynamic DB YT SK LJ --param T --range 1 100 10 ) &

#--dataset PR EN FB DB YT --dynamic DB YT