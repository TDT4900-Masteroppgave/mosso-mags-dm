./run.sh sweep --runs 3 --algorithm sm --group insertion dynamic --param h --range 5 50 5
./run.sh sweep --runs 3 --algorithm top-b --group insertion dynamic --param b --range 1 10 1
./run.sh sweep --runs 3 --algorithm cap --group insertion dynamic --param cap --range 100 505 45
./run.sh sweep --runs 3 --algorithm ds_thr --group insertion dynamic --param thr_end --range 0.1 1.0 0.1
./run.sh sweep --runs 3 --algorithm ds_thr --group insertion dynamic --param T --range 10 100 10