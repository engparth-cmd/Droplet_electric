"""Flow-only runs for Fig. 3(a): droplet shapes are identical to the full runs (flow -> ions -> field
is one-way coupled), but no ions/electrostatics are solved, so this is ~2x faster.

usage: python3 run_fig3a.py tag v0 '{"name": {cfg overrides}, ...}'
writes fig3a_<tag>_<name>.pkl in the format plot_fig3a.py reads, and prints shape metrics.
"""
import sys
import json
import time
import pickle
import numpy as np
from lbm_droplet_fig3 import DropletSim, DEFAULT_CFG

tag, v0, variants = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])


def shape(sim, rho):
    """length along the slope, max thickness, and rear/front thickness of the liquid touching
    the plate (plate frame, mm)"""
    c = sim.cfg
    liq = rho > 0.5 * (c["rho_l"] + c["rho_v"])
    liq[sim.solid] = False
    cols = np.where(liq.any(axis=0))[0]
    if len(cols) == 0:
        return 0, 0, 0, 0
    h = np.array([np.where(liq[:, i])[0].max() - sim.ytop + 1 if liq[:, i].any() else 0 for i in cols])
    L = len(cols)
    q = max(L // 4, 1)
    dx = 2.0 / c["R"]
    return L * dx, h.max() * dx, h[:q].mean() * dx, h[-q:].mean() * dx


for name, ch in variants.items():
    cfg = dict(DEFAULT_CFG)
    cfg.update(ch)
    sim = DropletSim(cfg)
    sim.equilibrate(cfg["n_equil"])
    sim.kick(v0)
    gx, gy = sim.g_lat * np.sin(sim.alpha), -sim.g_lat * np.cos(sim.alpha)
    snaps, hist = [], dict(t=[], contact=[], xcom=[], L=[], H=[], rear=[], front=[])
    had, zero, t0 = False, 0, time.time()
    for step in range(cfg.get("nsteps", 14000)):
        if step % 50 == 0:
            L, H, rear, front = shape(sim, sim.rho)
            cl = sim.contact_length()
            hist["t"].append(step * sim.t_scale); hist["contact"].append(cl); hist["xcom"].append(sim.com_x())
            hist["L"].append(L); hist["H"].append(H); hist["rear"].append(rear); hist["front"].append(front)
        cl_now = hist["contact"][-1]
        near_release = had and cl_now < 2 * cfg["R"] and zero < 400     # droplet leaving: finer frames
        if step % 250 == 0 or (step < 2000 and step % 100 == 0) or (near_release and step % 50 == 0):
            f32 = np.float32
            snaps.append(dict(step=step, t=step * sim.t_scale, rho=sim.rho.astype(f32),
                              ux=sim.ux.astype(f32), uy=sim.uy.astype(f32), contact=sim.contact_length()))
        sim._flow_step(gx, gy)
        if step % 500 == 0 and not np.isfinite(sim.rho).all():
            print(name, "BLEW UP at step", step, flush=True)
            break
        cl = sim.contact_length()
        had = had or cl > 0
        zero = zero + 1 if (had and cl == 0) else 0
        if zero > 900:
            break
    t = np.array(hist["t"]); c = np.array(hist["contact"])
    x = np.array(hist["xcom"]) * 2.0 / cfg["R"] * 1e-3
    first = np.where(c > 0)[0][0] if (c > 0).any() else 0
    gone = np.where((c == 0) & (np.arange(len(c)) > first))[0]
    leave = gone[0] if len(gone) else len(c) - 1
    vx = np.gradient(x, t)
    print(f"{name} ({time.time() - t0:.0f}s): contact at {t[first]*1e3:.0f} ms, leaves at {t[leave]*1e3:.0f} ms", flush=True)
    for frac in (0.25, 0.5, 0.75, 0.95):
        i = first + int(frac * (leave - first))
        print(f"   t={t[i]*1e3:5.0f} ms  slide speed {vx[i]:.2f} m/s  length {hist['L'][i]:.1f} mm  "
              f"thickness {hist['H'][i]:.1f} mm (rear {hist['rear'][i]:.1f}, front {hist['front'][i]:.1f})  "
              f"L/H {hist['L'][i]/max(hist['H'][i],1e-9):.1f}", flush=True)
    pickle.dump(dict(cfg=cfg, results={v0: dict(snaps=snaps, hist=hist)}),
                open(f"fig3a_{tag}_{name}.pkl", "wb"))
print("DONE", flush=True)
