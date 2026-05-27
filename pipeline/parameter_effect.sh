#!/bin/bash

./run.sh sweep --algorithm MoSSo --dataset PR EN FB DB YT --dynamic DB YT --param c --range 10 240 10 &
sleep 2

./run.sh sweep --algorithm MoSSo --dataset PR EN FB DB YT --dynamic DB YT --param e --range 1 10 1 &
sleep 2

./run.sh sweep --algorithm sm --dataset PR EN FB DB YT --dynamic DB YT --param h --range 5 50 5 &
sleep 2

./run.sh sweep --algorithm sm --dataset PR EN FB DB YT --dynamic DB YT --param h --range 1 1 1 &
sleep 2

./run.sh sweep --algorithm sm_thr --dataset PR EN FB DB YT --dynamic DB YT --param thr --range 0.0 1.0 0.1 &
sleep 2

./run.sh sweep --algorithm top-b --dataset PR EN FB DB YT --dynamic DB YT --param b --range 1 10 1 &
sleep 2

./run.sh sweep --algorithm cap --dataset PR EN FB DB YT --dynamic DB YT --param cap --range 0 120 10 &
sleep 2

./run.sh sweep --algorithm ds_thr --dataset PR EN FB DB YT --dynamic DB YT --param T --range 10 100 10 &
sleep 2

./run.sh sweep --algorithm ds_thr --dataset PR EN FB DB YT --dynamic DB YT --param T --range 1 1 1 &