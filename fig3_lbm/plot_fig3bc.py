"""Fig. 3(b) and 3(c): electrode charge and current from lbm_droplet_fig3.py, compared with the
paper's curves (digitised from the published figure, in paper_data/).

usage: python3 plot_fig3bc.py [fig3_results.pkl]
writes fig3b_compare.png and fig3c_compare.png and prints the comparison numbers.
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
fn = sys.argv[1] if len(sys.argv) > 1 else "fig3_results.pkl"
res = pickle.load(open(fn, "rb"))["results"]
PB = np.load(os.path.join(HERE, "paper_data", "paper_b_digitized.npy"), allow_pickle=True).item()
PC = np.load(os.path.join(HERE, "paper_data", "paper_c_digitized.npy"), allow_pickle=True).item()
vs = [v for v in (0.0, 0.1, 0.2, 0.3) if v in res]          # whichever velocities were simulated
col = {0.0: "tab:red", 0.1: "tab:orange", 0.2: "tab:blue", 0.3: "k"}


def smooth(y, k):
    return np.convolve(np.pad(y, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


# ---------------------------------------------------------------- data
def paper_charge(v):
    d = PB[f"{v:.1f}"]
    t = np.concatenate([[0.0], d["inset"][:, 0], d["main"][:, 0]])
    q = np.concatenate([[237.30], d["inset"][:, 1], d["main"][:, 1]])  # all curves start at 237.3 nC
    o = np.argsort(t)
    return t[o], smooth(q[o], 5)


def paper_current(v):
    a = PC[f"{v:.1f}"]
    o = np.argsort(a[:, 0])
    return a[o, 0], smooth(a[o, 1], 5)


def our_charge(v):
    h = res[v]["hist"]
    return np.array(h["t"]), np.array(h["Q"]) * 243.0       # scaled: no droplet = 243 nC


def our_current(v):
    t, Q = our_charge(v)
    return t, smooth(np.gradient(Q, t), 3)                   # dQ/dt in nC/s = nA


def peaks(t, i):
    """landing (negative) peak, release (positive) peak and its full width at half maximum"""
    e = t < 0.07
    k1 = np.where(e)[0][i[e].argmin()]
    l = t > 0.08
    k2 = np.where(l)[0][i[l].argmax()]
    a = b = k2
    while a > 0 and i[a] > i[k2] / 2:
        a -= 1
    while b < len(i) - 1 and i[b] > i[k2] / 2:
        b += 1
    return i[k1], t[k1] * 1e3, i[k2], t[k2] * 1e3, (t[b] - t[a]) * 1e3


def levels(t, q, t_flat=0.06):
    return q[0], q[np.argmin(abs(t - t_flat))], q[-1]


# ---------------------------------------------------------------- numbers
print("Fig. 3(c) current                     paper                       |   this work")
print(" v0   landing peak   release peak  width | landing peak   release peak  width")
P_pk, O_pk = {}, {}
for v in vs:
    P_pk[v] = peaks(*paper_current(v))
    O_pk[v] = peaks(*our_current(v))
    p, o = P_pk[v], O_pk[v]
    print(f" {v:.1f}  {p[0]:6.1f} nA @{p[1]:4.0f}  {p[2]:5.0f} nA @{p[3]:4.0f}  {p[4]:3.0f} ms |"
          f" {o[0]:6.1f} nA @{o[1]:4.0f}  {o[2]:5.0f} nA @{o[3]:4.0f}  {o[4]:3.0f} ms")
print("\nFig. 3(b) electrode charge (nC)   paper: start / sliding / end   |  this work")
for v in vs:
    p = levels(*paper_charge(v), t_flat=0.07)
    o = levels(*our_charge(v))
    print(f" v0={v:.1f}   {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}   |   {o[0]:.2f} / {o[1]:.2f} / {o[2]:.2f}")

# ---------------------------------------------------------------- figure (b)
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
af = axs.ravel()
for a, v in zip(af[:4], vs):
    a.plot(*paper_charge(v), "--", color="gray", lw=2.2, label="paper (digitised)")
    a.plot(*our_charge(v), color=col[v], lw=2, label="this work (LBM)")
    a.set_xlim(0, 0.19); a.set_ylim(234, 243.5); a.grid(alpha=0.25)
    a.set_title(f"v0 = {v:.1f} m/s"); a.set_xlabel("Time (s)"); a.set_ylabel("Electrode charge (nC)")
    a.legend(frameon=False, loc="lower right", fontsize=9)
a = af[4]
for v in vs:
    tp, qp = paper_charge(v); to, qo = our_charge(v)
    a.plot(tp[tp < 0.045], qp[tp < 0.045], "--", color=col[v])
    a.plot(to[to < 0.045], qo[to < 0.045], color=col[v], label=f"v0 = {v:.1f}")
a.set_xlim(0, 0.045); a.set_ylim(235.5, 238.0); a.grid(alpha=0.25); a.legend(frameon=False, fontsize=9)
a.set_title("Landing, first 45 ms (dashed = paper, solid = this work)")
a.set_xlabel("Time (s)"); a.set_ylabel("Electrode charge (nC)")
a = af[5]
a.plot(vs, [P_pk[v][3] for v in vs], "s--", color="tab:purple", label="paper")
a.plot(vs, [O_pk[v][3] for v in vs], "o-", color="tab:green", label="this work")
a.set_xlabel("Initial velocity (m/s)"); a.set_ylabel("Time of the release current peak (ms)")
a.set_title("Release timing"); a.legend(frameon=False); a.grid(alpha=0.3)
for a in af[len(vs):4]:
    a.axis("off")
plt.tight_layout()
plt.savefig("fig3b_compare.png", dpi=130)
plt.close()

# ---------------------------------------------------------------- figure (c)
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
af = axs.ravel()
for a, v in zip(af[:4], vs):
    a.plot(*paper_current(v), "--", color="gray", lw=2.2, label="paper (digitised)")
    a.plot(*our_current(v), color=col[v], lw=2, label="this work (dQ/dt)")
    a.axhline(0, color="k", lw=0.5); a.set_xlim(0, 0.19); a.set_ylim(-110, 340); a.grid(alpha=0.25)
    a.set_title(f"v0 = {v:.1f} m/s"); a.set_xlabel("Time (s)"); a.set_ylabel("Current (nA)")
    a.legend(frameon=False, loc="upper left", fontsize=9)
for a, j, title in ((af[4], 0, "First (negative) peak, at landing"), (af[5], 2, "Second (positive) peak, at release")):
    a.plot(vs, [P_pk[v][j] for v in vs], "s--", color="tab:purple", label="paper")
    a.plot(vs, [O_pk[v][j] for v in vs], "o-", color="tab:green", label="this work")
    a.set_title(title); a.set_xlabel("Initial velocity (m/s)"); a.set_ylabel("Current (nA)")
    a.legend(frameon=False); a.grid(alpha=0.3)
for a in af[len(vs):4]:
    a.axis("off")
plt.tight_layout()
plt.savefig("fig3c_compare.png", dpi=130)
plt.close()
print("\nsaved fig3b_compare.png and fig3c_compare.png")
