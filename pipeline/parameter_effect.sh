./run.sh sweep --runs 3 --algorithm sm --group insertion dynamic --param h --range 1 51 10
./run.sh sweep --runs 3 --algorithm top-b --group insertion dynamic --param b --range 1 11 2
./run.sh sweep --runs 3 --algorithm cap --group insertion dynamic --param cap --range 100 5100 100
./run.sh sweep --runs 3 --algorithm ds_thr --group insertion dynamic --param thr_end --range 0 1 0.10
./run.sh sweep --runs 3 --algorithm ds_thr --group insertion dynamic --param T --range 1 51 10