# LBM re-creation of Fig. 3 (a, b, c) — Monluc et al., Nano Energy 146 (2025) 111536

A 2D lattice-Boltzmann model of a charged water droplet that falls on, slides down and leaves an
inclined PTFE/copper plate, with the charge on the copper electrode (Fig. 3b) and the current
(Fig. 3c), compared with the paper.

## Which file do I run?

**Everything at once (about 45 minutes):**

```bash
bash run_all.sh
```

**Or step by step** (same thing, one command at a time):

| Step | Command | Time | Makes |
|---|---|---|---|
| 1 | `python3 lbm_droplet_fig3.py` | ~40 min | `fig3_results.pkl` (all 4 velocities) |
| 2 | `python3 run_fig3a.py 0.0` | ~4 min | `fig3a_v0.0.pkl` (droplet shapes) |
| 3 | `python3 plot_fig3a.py fig3a_v0.0.pkl 0.0` | ~30 s | `fig3a_compare.png` (Fig. 3a vs paper) |
| 4 | `python3 plot_fig3bc.py fig3_results.pkl` | ~30 s | `fig3b_compare.png`, `fig3c_compare.png` + numbers |

Steps 1 and 2 are independent: run step 2 alone if you only want Fig. 3(a), or steps 1 and 4 if
you only want Fig. 3(b),(c). Plotting (steps 3, 4) can be repeated without re-running the simulation.

## Install first (Ubuntu)

```bash
pip install numpy scipy numba matplotlib pillow
```

Python 3.9 or newer. The first run takes ~30 s extra while numba compiles the kernels.
Needs about 2-3 GB of RAM and ~0.5 GB of disk for `fig3_results.pkl`.

## What each file does

| File | Role |
|---|---|
| `lbm_droplet_fig3.py` | **The model.** Droplet flow (Shan-Chen LBM, MRT collision), wetting with a contact-angle window, Navier slip on the PTFE, ion transport (Nernst-Planck), electric field (Poisson), electrode charge. All settings are in `DEFAULT_CFG` at the bottom. |
| `run_fig3a.py` | Runs only the droplet flow (no charges), saving many frames for Fig. 3(a). The droplet motion is identical to step 1, only faster to compute. |
| `plot_fig3a.py` | Draws Fig. 3(a) next to the paper's panel, the way the paper does: fill = water fraction (navy air, red water), arrows = velocity coloured by speed (0-0.6 m/s). |
| `plot_fig3bc.py` | Draws Fig. 3(b) and 3(c) against the paper's curves and prints the comparison table. |
| `run_all.sh` | Runs the four steps in order. |
| `paper_data/` | The paper's Fig. 3(b),(c) curves digitised from the published figure, and its four Fig. 3(a) phase images (used only for the side-by-side comparison). |

## Useful variations

* One velocity only: `python3 lbm_droplet_fig3.py --v 0.0`
* Fig. 3(a) for another velocity: `python3 run_fig3a.py 0.3` then `python3 plot_fig3a.py fig3a_v0.3.pkl 0.3 fig3a_v03.png`
* Change a setting without editing the code: `python3 run_fig3a.py 0.0 test.pkl '{"theta_r": 30}'`
  (any key of `DEFAULT_CFG`), or for the full model put the changes in a JSON file and use
  `python3 lbm_droplet_fig3.py --cfg my_settings.json`.

## Fitted settings (not predictions)

These were adjusted to match the paper's figure: release height (`gap`, 2.2 mm), start position on
the plate (`x0`), receding-angle limit of the wetting window (`theta_r`, 20 deg), wall friction
(`r_bb`), viscosity (`tau`), droplet charge (`drop_charge_frac`), ion migration (`mig_ratio`) and the
interface pull (`conf_band`). Known limits: the liquid/gas density ratio is ~12 (real water/air ~800)
and the liquid is ~5x more viscous than water, so the sliding droplet is shorter and slower than the
paper's (about 8 mm and 0.25 m/s against 10 mm and 0.45 m/s in phase III).
