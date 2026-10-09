#!/usr/bin/env bash
# Reproduce Fig. 3(a), (b), (c) of Monluc et al., Nano Energy 146 (2025) 111536 with the LBM model.
# Run from anywhere:   bash run_all.sh
# Total time: about 45 minutes on one CPU core.
set -e
cd "$(dirname "$0")"

echo "== 1/4  Simulation for Fig. 3(b),(c): v0 = 0, 0.1, 0.2, 0.3 m/s (about 40 min) =="
python3 lbm_droplet_fig3.py --v 0.0 0.1 0.2 0.3 --out fig3_results.pkl

echo "== 2/4  Droplet shapes for Fig. 3(a): flow-only run, v0 = 0 (about 4 min) =="
python3 run_fig3a.py 0.0 fig3a_v0.0.pkl

echo "== 3/4  Fig. 3(a) picture =="
python3 plot_fig3a.py fig3a_v0.0.pkl 0.0 fig3a_compare.png

echo "== 4/4  Fig. 3(b),(c) pictures and numbers =="
python3 plot_fig3bc.py fig3_results.pkl

echo "Done: fig3a_compare.png, fig3b_compare.png, fig3c_compare.png"
