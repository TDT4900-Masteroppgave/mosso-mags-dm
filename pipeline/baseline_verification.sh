./run.sh speed --runs 3 --group insertion dynamic --algorithm kdd20-mosso mags mags_dm
./run.sh cc --runs 3 --group insertion dynamic  --algorithm kdd20-mosso mags mags_dm
./run.sh scale --runs 3 --group insertion dynamic --algorithm kdd20-mosso mags mags_dm
./run.sh benchmark --runs 3 --group mags_small --algorithm mags mags_dm
./run.sh benchmark --runs 3 --group mags_large --algorithm mags mags_dm
./run.sh benchmark --runs 3 --group mags_small --algorithm para_mags para_mags_dm --p 30
./run.sh benchmark --runs 3 --group mags_large --algorithm para_mags para_mags_dm --p 30