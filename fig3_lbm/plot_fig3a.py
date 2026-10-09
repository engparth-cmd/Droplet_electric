"""Fig. 3(a)-style droplet phases from lbm_droplet_fig3.py results, next to the paper's panel.

Drawn the way the paper's COMSOL figure is drawn:
  * background colour = water volume fraction (air 0 = navy, water 1 = dark red, 'jet' colours),
    so the diffuse interface shows as a thin rainbow band,
  * arrows inside the droplet = velocity, coloured by speed on a 0-0.6 m/s scale (the paper's
    colour bar 'v (m/s)' belongs to the arrows),
  * PTFE surface = thick black line, plate tilted down to the right.

usage: python3 plot_fig3a.py fig3a_v0.0.pkl [velocity] [out.png]
  (the .pkl comes from run_fig3a.py, or from lbm_droplet_fig3.py)
The paper's four phase images are read from paper_data/paper_phase_1..4.png for the left column;
without them only the simulation column is drawn.
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = [os.path.join(HERE, "paper_data", f"paper_phase_{k}.png") for k in range(1, 5)]

fn = sys.argv[1] if len(sys.argv) > 1 else "fig3a_v0.0.pkl"
v_show = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
out = sys.argv[3] if len(sys.argv) > 3 else "fig3a_compare.png"
D = pickle.load(open(fn, "rb"))
cfg, res = D["cfg"], D["results"]
snaps = res[v_show]["snaps"]

# lattice -> physical velocity (same Weber-number scaling as the simulation)
v_scale = np.sqrt((cfg["sigma_lat"] / (cfg["rho_l"] * 2 * cfg["R"])) / (0.072 / (1000.0 * 2 * 2e-3)))
dx_mm = 2.0 / cfg["R"]                     # droplet radius = 2 mm
alpha = np.deg2rad(cfg["alpha_deg"])
ca, sa = np.cos(alpha), np.sin(alpha)
rho_l, rho_v = cfg["rho_l"], cfg["rho_v"]
ny, nx, Lp, ytop, R = cfg["ny"], cfg["nx"], cfg["Lp"], cfg["ytop"], cfg["R"]


def pick_frames(snaps):
    """I: first frame touching the plate (the paper's phase I droplet already touches);
    II: early impact = first frame whose wetted base reaches 1.3 droplet diameters (top still raised);
    III: most stretched frame while sliding; IV: separation, tail still touching the plate end."""
    t = np.array([s["t"] for s in snaps])
    c = np.array([s["contact"] for s in snaps])
    first = np.where(c > 0)[0][0]
    i1 = first
    spread = np.where((c >= 1.3 * 2 * R) & (np.arange(len(c)) > first))[0]
    if len(spread) and t[spread[0]] <= t[first] + 0.03:
        i2 = spread[0]
    else:
        w = np.where((t >= t[first]) & (t <= t[first] + 0.03))[0]
        i2 = w[np.argmax(c[w])]
    gone = np.where((c == 0) & (np.arange(len(c)) > first))[0]
    leave = gone[0] if len(gone) else len(c) - 1
    sl = np.where((t >= t[first] + 0.04) & (t <= t[leave] - 0.01))[0]
    i3 = sl[np.argmax(c[sl])] if len(sl) else (i2 + leave) // 2
    # IV: separation = the droplet is mostly past the plate end but its tail still touches the edge
    # (wetted length under a quarter of a diameter), as in the paper's phase IV
    tail = np.where((c > 0) & (c <= 0.25 * 2 * R) & (np.arange(len(c)) > first) & (t >= t[leave] - 0.02)
                    & (np.arange(len(c)) < leave))[0]
    if len(tail):
        i4 = tail[-1]
    else:
        after = np.where(t >= t[leave])[0]
        i4 = after[0] if len(after) else len(c) - 1
    return [i1, i2, i3, i4]


frames = pick_frames(snaps)
if len(sys.argv) > 4:                       # optional explicit frame times in ms: "t1,t2,t3,t4"
    want = [float(x) / 1e3 for x in sys.argv[4].split(",")]
    tt = np.array([s["t"] for s in snaps])
    frames = [int(np.argmin(abs(tt - w))) for w in want]
names = ["I", "II", "III", "IV"]
yy, xx = np.mgrid[0:ny, 0:nx]
X = (xx * ca + yy * sa) * dx_mm            # lab frame, mm
Y = (-xx * sa + yy * ca) * dx_mm
solid = np.zeros((ny, nx), bool)
solid[:ytop, :Lp] = True
norm = Normalize(0.0, 0.6)
jet = plt.get_cmap("jet")

have_paper = all(os.path.exists(p) for p in PAPER)
ncol = 2 if have_paper else 1
fig, axs = plt.subplots(4, ncol, figsize=(4.8 * ncol + 1.3, 12.5), squeeze=False)
for k, fi in enumerate(frames):
    if have_paper:
        axs[k, 0].imshow(Image.open(PAPER[k]))
        axs[k, 0].axis("off")
    a = axs[k, ncol - 1]
    s = snaps[fi]
    frac = np.clip((s["rho"] - rho_v) / (rho_l - rho_v), 0, 1)
    frac[solid] = 0.0                      # plate drawn like the air (navy), as in the paper
    a.pcolormesh(X, Y, frac, cmap=jet, vmin=0, vmax=1, shading="gouraud", rasterized=True)
    xs = np.arange(Lp)
    a.plot((xs * ca + (ytop - 0.5) * sa) * dx_mm, (-xs * sa + (ytop - 0.5) * ca) * dx_mm, color="k", lw=4,
           solid_capstyle="butt")
    liq = (frac > 0.6) & ~solid
    st = 4
    sub = np.zeros_like(liq)
    sub[::st, ::st] = True
    jj, ii = np.where(liq & sub)
    ux, uy = s["ux"][jj, ii], s["uy"][jj, ii]
    vx_lab, vy_lab = ux * ca + uy * sa, -ux * sa + uy * ca
    spd = np.hypot(ux, uy) / v_scale
    a.quiver(X[jj, ii], Y[jj, ii], vx_lab, vy_lab, spd, cmap=jet, norm=norm, width=0.0045,
             scale=v_scale * 0.6 * 22, scale_units="width", headwidth=3.5, headlength=4)
    xc, yc = X[liq].mean(), Y[liq].mean()
    a.set_xlim(xc - 5.3, xc + 5.3)
    a.set_ylim(yc - 3.5, yc + 3.5)
    a.set_facecolor(jet(0.0))
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.text(0.03, 0.06, f"{names[k]}   t = {s['t'] * 1e3:.0f} ms", transform=a.transAxes, color="w",
           fontsize=10, fontweight="bold")
if have_paper:
    axs[0, 0].set_title("Paper, Fig. 3(a) (COMSOL)")
axs[0, ncol - 1].set_title(f"This work (LBM), v0 = {v_show:.1f} m/s")
sm = plt.cm.ScalarMappable(norm=norm, cmap=jet)
cb = fig.colorbar(sm, ax=axs[:, ncol - 1], shrink=0.45, pad=0.03)
cb.set_label("arrows: v (m/s)    |    fill: water fraction (navy = air, red = water)")
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out, "frames at", [round(snaps[f]["t"] * 1e3) for f in frames], "ms")
