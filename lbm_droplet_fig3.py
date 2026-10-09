"""
Lattice-Boltzmann model of a charged droplet falling on, sliding down and leaving
an inclined PTFE/copper plate (2D).  Re-creation of Fig. 3(a), (b), (c) of
Monluc et al., Nano Energy 146 (2025) 111536.

Physics (same building blocks as the paper, different numerical method):
  1. Two-phase flow ............ D2Q9 Shan-Chen pseudopotential LBM, MRT collision with
                                 exact-difference forcing (density ratio ~12)
  2. Wetting (contact angle) ... geometric wetting condition (virtual wall density) with a
                                 contact-angle window: advancing angle 105 deg (Table S1); inside
                                 the window the contact line is carried by the slip flow, as with
                                 the slowly relaxing wetted-wall condition of the paper's
                                 phase-field model (wall_model="force": constant Gw instead)
  3. Wall slip ................. Navier (partial) slip on the PTFE surface, done as a mix of
                                 bounce-back and specular reflection (paper, Sec. 2.7)
  4. Ion transport ............. D2Q5 advection-diffusion-migration LBM (Nernst-Planck),
                                 no-flux at the water/air interface in diffuse-domain form
                                 (drift D grad(m)/m, equilibrium c ~ water fraction m)
  5. Electrostatics ............ Poisson equation on a padded grid (grounded far boundary)
  6. Electrode charge Q(t) ..... Gauss law on the grounded copper film; current = dQ/dt

Simplifications:
  * one-way coupling: flow -> ions -> field. Electric force on the fluid is neglected
    (the paper itself finds it negligible for this case).
  * the inclined plate is simulated in the plate frame; gravity is rotated instead
    (g_x = g sin(alpha) along the slope, g_y = -g cos(alpha) normal to it).
  * everything is in lattice units, mapped to physical units through the Weber number
    (velocity), Bond number (gravity) and the capillary time.
  * the Ohnesorge number is ~5x larger than for real water and the liquid/gas density ratio
    is ~12 instead of ~800 (stability limits of the pseudopotential LBM).
  * charge amount, ion migration strength, release height and start position are fitted to
    the paper's Fig. 3(b),(c); they are not predictions.
"""
import numpy as np
import numba as nb
import time
import json
import argparse
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, splu

# ----------------------------------------------------------------------------------
# lattice definitions
# ----------------------------------------------------------------------------------
EX9 = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
EY9 = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

EX5 = np.array([0, 1, 0, -1, 0], dtype=np.int64)
EY5 = np.array([0, 0, 1, 0, -1], dtype=np.int64)
W5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
OPP5 = np.array([0, 3, 4, 1, 2], dtype=np.int64)


# ----------------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------------
@nb.njit(cache=True)
def k_macros(f, rho, jx, jy, psi, solid):
    ny, nx = rho.shape
    for j in range(ny):
        for i in range(nx):
            if solid[j, i]:
                rho[j, i] = 0.0
                jx[j, i] = 0.0
                jy[j, i] = 0.0
                psi[j, i] = 0.0
                continue
            r = 0.0
            sx = 0.0
            sy = 0.0
            for k in range(9):
                v = f[k, j, i]
                r += v
                sx += v * EX9[k]
                sy += v * EY9[k]
            rho[j, i] = r
            jx[j, i] = sx
            jy[j, i] = sy
            psi[j, i] = 1.0 - np.exp(-r)  # Shan-Chen psi, rho0 = 1


@nb.njit(cache=True)
def k_collide(f, ftmp, rho, jx, jy, ux, uy, psi, solid, G, Gwm, gx, gy, tau):
    ny, nx = rho.shape
    for j in range(ny):
        for i in range(nx):
            if solid[j, i]:
                continue
            r = rho[j, i]
            p = psi[j, i]
            fx = 0.0
            fy = 0.0
            for k in range(1, 9):
                jj = j + EY9[k]
                ii = i + EX9[k]
                if jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                    c = 0.0
                elif solid[jj, ii]:
                    c = -Gwm[jj, ii] * p * W9[k]     # wall interaction depends on the wall node
                else:
                    c = -G * p * psi[jj, ii] * W9[k]
                fx += c * EX9[k]
                fy += c * EY9[k]
            fx += r * gx
            fy += r * gy
            ux[j, i] = (jx[j, i] + 0.5 * fx) / r
            uy[j, i] = (jy[j, i] + 0.5 * fy) / r
            vx = jx[j, i] / r + tau * fx / r
            vy = jy[j, i] / r + tau * fy / r
            usq = vx * vx + vy * vy
            for k in range(9):
                eu = EX9[k] * vx + EY9[k] * vy
                feq = W9[k] * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
                ftmp[k, j, i] = f[k, j, i] - (f[k, j, i] - feq) / tau


# D2Q9 MRT (Lallemand & Luo 2000). Moments: rho, e, eps, jx, qx, jy, qy, pxx, pxy.
# Velocity ordering is the same as EX9/EY9 above.
M9 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
               [-4, -1, -1, -1, -1, 2, 2, 2, 2],
               [4, -2, -2, -2, -2, 1, 1, 1, 1],
               [0, 1, 0, -1, 0, 1, -1, -1, 1],
               [0, -2, 0, 2, 0, 1, -1, -1, 1],
               [0, 0, 1, 0, -1, 1, 1, -1, -1],
               [0, 0, -2, 0, 2, 1, 1, -1, -1],
               [0, 1, -1, 1, -1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=np.float64)
M9INV = np.linalg.inv(M9)


@nb.njit(cache=True)
def k_collide_mrt(f, ftmp, rho, jx, jy, ux, uy, psi, solid, G, Gwm, gx, gy, S):
    """MRT collision with the exact-difference forcing (Kupershtokh 2009).
    More stable than BGK at low viscosity, and the liquid/gas densities do not depend on the
    relaxation time. S = relaxation rates of the 9 moments (S[7] = S[8] = 1/tau sets viscosity)."""
    ny, nx = rho.shape
    m = np.empty(9)
    for j in range(ny):
        for i in range(nx):
            if solid[j, i]:
                continue
            r = rho[j, i]
            p = psi[j, i]
            fx = 0.0
            fy = 0.0
            for k in range(1, 9):
                jj = j + EY9[k]
                ii = i + EX9[k]
                if jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                    c = 0.0
                elif solid[jj, ii]:
                    c = -Gwm[jj, ii] * p * W9[k]
                else:
                    c = -G * p * psi[jj, ii] * W9[k]
                fx += c * EX9[k]
                fy += c * EY9[k]
            fx += r * gx
            fy += r * gy
            ux[j, i] = (jx[j, i] + 0.5 * fx) / r      # physical velocity
            uy[j, i] = (jy[j, i] + 0.5 * fy) / r
            u0 = jx[j, i] / r
            v0 = jy[j, i] / r
            # moments
            for a in range(9):
                s = 0.0
                for k in range(9):
                    s += M9[a, k] * f[k, j, i]
                m[a] = s
            usq = u0 * u0 + v0 * v0
            meq0 = r
            meq1 = r * (-2.0 + 3.0 * usq)
            meq2 = r * (1.0 - 3.0 * usq)
            meq3 = r * u0
            meq4 = -r * u0
            meq5 = r * v0
            meq6 = -r * v0
            meq7 = r * (u0 * u0 - v0 * v0)
            meq8 = r * u0 * v0
            m[0] -= S[0] * (m[0] - meq0)
            m[1] -= S[1] * (m[1] - meq1)
            m[2] -= S[2] * (m[2] - meq2)
            m[3] -= S[3] * (m[3] - meq3)
            m[4] -= S[4] * (m[4] - meq4)
            m[5] -= S[5] * (m[5] - meq5)
            m[6] -= S[6] * (m[6] - meq6)
            m[7] -= S[7] * (m[7] - meq7)
            m[8] -= S[8] * (m[8] - meq8)
            # exact-difference force term: feq(u0 + F/rho) - feq(u0)
            u1 = u0 + fx / r
            v1 = v0 + fy / r
            usq1 = u1 * u1 + v1 * v1
            for k in range(9):
                s = 0.0
                for a in range(9):
                    s += M9INV[k, a] * m[a]
                eu0 = EX9[k] * u0 + EY9[k] * v0
                eu1 = EX9[k] * u1 + EY9[k] * v1
                d = W9[k] * r * ((3.0 * eu1 + 4.5 * eu1 * eu1 - 1.5 * usq1)
                                 - (3.0 * eu0 + 4.5 * eu0 * eu0 - 1.5 * usq))
                ftmp[k, j, i] = s + d


@nb.njit(cache=True)
def k_stream9(f, ftmp, solid, slip_row, r_bb):
    """Streaming with walls.
    Ordinary walls: full bounce-back (no-slip).
    PTFE top surface (solid row = slip_row): Navier slip, done as a mix of bounce-back
    (fraction r_bb) and specular reflection (fraction 1 - r_bb). r_bb = 1 is no-slip,
    r_bb = 0 is free slip; in between gives a finite slip length, as in the paper (Sec. 2.7)."""
    ny, nx = solid.shape
    for j in range(ny):
        for i in range(nx):
            if solid[j, i]:
                continue
            for k in range(9):
                js = j - EY9[k]
                is_ = i - EX9[k]
                if js < 0 or js >= ny or is_ < 0 or is_ >= nx:
                    f[k, j, i] = ftmp[OPP9[k], j, i]
                elif solid[js, is_]:
                    bb = ftmp[OPP9[k], j, i]
                    # specular partner for an upward-moving diagonal leaving the PTFE surface:
                    # (ex, +1) at (j, i) came from (ex, -1) leaving (j, i - ex)
                    if js == slip_row and EY9[k] == 1 and EX9[k] != 0 and r_bb < 1.0:
                        ii = i - EX9[k]
                        if 0 <= ii < nx and not solid[j, ii]:
                            kp = 8 if EX9[k] == 1 else 7
                            f[k, j, i] = r_bb * bb + (1.0 - r_bb) * ftmp[kp, j, ii]
                        else:
                            f[k, j, i] = bb
                    else:
                        f[k, j, i] = bb
                else:
                    f[k, j, i] = ftmp[k, js, is_]


@nb.njit(cache=True)
def k_ion_collide(g, gtmp, c, vx, vy, tauc, solid):
    ny, nx = c.shape
    for j in range(ny):
        for i in range(nx):
            if solid[j, i]:
                c[j, i] = 0.0
                continue
            s = 0.0
            for k in range(5):
                s += g[k, j, i]
            c[j, i] = s
            for k in range(5):
                eu = EX5[k] * vx[j, i] + EY5[k] * vy[j, i]
                geq = W5[k] * s * (1.0 + 3.0 * eu)
                gtmp[k, j, i] = g[k, j, i] - (g[k, j, i] - geq) / tauc[j, i]


@nb.njit(cache=True)
def k_ion_stream(g, gtmp, solid):
    ny, nx = solid.shape
    for j in range(ny):
        for i in range(nx):
            if solid[j, i]:
                continue
            for k in range(5):
                js = j - EY5[k]
                is_ = i - EX5[k]
                if js < 0 or js >= ny or is_ < 0 or is_ >= nx:
                    g[k, j, i] = gtmp[OPP5[k], j, i]
                elif solid[js, is_]:
                    g[k, j, i] = gtmp[OPP5[k], j, i]
                else:
                    g[k, j, i] = gtmp[k, js, is_]


# ----------------------------------------------------------------------------------
# Poisson solver  div( eps grad psi ) = -rho_q ,  psi = 0 on copper and outer box
# ----------------------------------------------------------------------------------
def solve_poisson(eps, rho_q, dirichlet):
    ny, nx = eps.shape
    free = ~dirichlet
    idx = -np.ones((ny, nx), dtype=np.int64)
    n = int(free.sum())
    idx[free] = np.arange(n)
    jj, ii = np.nonzero(free)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    me = idx[jj, ii]
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        j2, i2 = jj + dj, ii + di
        a = 2 * eps[jj, ii] * eps[j2, i2] / (eps[jj, ii] + eps[j2, i2])
        diag -= a
        m = free[j2, i2]
        rows.append(me[m])
        cols.append(idx[j2, i2][m])
        vals.append(a[m])
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsc()
    if rho_q is None:                 # factorize only: return a reusable solver
        return splu(A), free
    psi = np.zeros((ny, nx))
    psi[free] = spsolve(A, -rho_q[free])
    return psi


def solve_poisson_lu(lu, free, rho_q):
    psi = np.zeros(rho_q.shape)
    psi[free] = lu.solve(-rho_q[free])
    return psi


# ----------------------------------------------------------------------------------
# simulation class
# ----------------------------------------------------------------------------------
class DropletSim:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg
        self.nx, self.ny = c["nx"], c["ny"]
        nx, ny = self.nx, self.ny
        self.R = c["R"]
        self.Lp = c["Lp"]              # plate length (nodes)
        self.ytop = c["ytop"]          # first fluid row above plate = ytop
        self.alpha = np.deg2rad(c["alpha_deg"])

        # ---- geometry (solid mask for the flow) ----
        solid = np.zeros((ny, nx), dtype=np.bool_)
        solid[: self.ytop, : self.Lp] = True       # plate block (copper + PTFE row on top)
        solid[0, :] = solid[-1, :] = True          # outer ring = walls
        solid[:, 0] = solid[:, -1] = True
        self.solid = solid
        # wall wetting map: the top PTFE surface gets Gw (contact angle ~106 deg); every other solid
        # node (end face of the plate, outer box) is made non-wetting so the droplet does not cling
        # to the end of the plate and can really separate (Fig. 3a, phase IV)
        self.Gw_map = np.full((ny, nx), c.get("Gw_far", 0.0))
        self.Gw_map[self.ytop - 1, : self.Lp] = c["Gw"]
        # contact-angle hysteresis by wetting memory: a PTFE node that the water has covered
        # (water fraction above it > wet_on) switches to the receding wall strength Gw_wet and keeps
        # it until the water has really left (fraction < wet_off). The advancing front therefore
        # sees dry PTFE (Gw, ~105 deg) and the rear sees wetted PTFE (Gw_wet, smaller angle), so the
        # rear contact line lags and the sliding droplet stretches, like the paper's phase-field
        # droplet whose contact line can only relax by slow diffusion.
        self.wet_mem = np.zeros(self.Lp, dtype=bool)
        self._hyst_on = "Gw_wet" in c and c["Gw_wet"] != c["Gw"]
        # geometric wetting with a contact-angle window (wall_model = "geometric"):
        # the PTFE wall layer gets a virtual density rho_w = rho_1 + tan(pi/2 - theta)*|d rho_1/dx|
        # (rho_1 = first fluid layer), and interacts with the fluid through the fluid-fluid G.
        # theta = the CURRENT local contact angle, clipped to [theta_r, theta_a]: inside the window
        # the wall exerts no capillary force on the contact line, which is then just carried by the
        # (slip) flow, as with the slowly relaxing wetted-wall condition of the paper's phase-field
        # model; outside it, the angle is pushed back to theta_r (receding) or theta_a (advancing).
        self._geo_on = c.get("wall_model", "force") == "geometric"

        # ---- fluid ----
        self.f = np.zeros((9, ny, nx))
        self.ftmp = np.zeros_like(self.f)
        self.rho = np.zeros((ny, nx))
        self.jx = np.zeros((ny, nx))
        self.jy = np.zeros((ny, nx))
        self.ux = np.zeros((ny, nx))
        self.uy = np.zeros((ny, nx))
        self.psi = np.zeros((ny, nx))
        self._init_droplet()

        # ---- scaling to physical units ----
        sig_l, rho_l = c["sigma_lat"], c["rho_l"]
        self.rho_phys, self.sig_phys, self.R_phys = 1000.0, 0.072, 2e-3
        t_cap_phys = np.sqrt(self.rho_phys * self.R_phys ** 3 / self.sig_phys)
        t_cap_lat = np.sqrt(rho_l * self.R ** 3 / sig_l)
        self.t_scale = t_cap_phys / t_cap_lat                       # s per lattice step
        Bo = self.rho_phys * 9.81 * self.R_phys ** 2 / self.sig_phys
        self.g_lat = Bo * sig_l / (rho_l * self.R ** 2)             # Bond-number matching
        # velocity scale from We: We = rho v^2 (2R)/sigma
        self.v_scale = np.sqrt((sig_l / (rho_l * 2 * self.R)) /
                               (self.sig_phys / (self.rho_phys * 2 * self.R_phys)))

        # ---- ions ----
        self.g = np.zeros((5, ny, nx))
        self.gtmp = np.zeros_like(self.g)
        self.cion = np.zeros((ny, nx))
        self.vx_ion = np.zeros((ny, nx))
        self.vy_ion = np.zeros((ny, nx))
        self.tauc = np.full((ny, nx), 1.0)

        # ---- electrostatics ----
        # The Poisson problem is solved on a larger grid than the flow: 'pad' extra nodes of air
        # on every side, so the grounded outer boundary is far from the droplet (the paper's
        # 'grounded open boundaries' sit far from the device).
        p = c.get("pad", 0)
        self.pad = p
        PNY, PNX = ny + 2 * p, nx + 2 * p
        dir_ = np.zeros((PNY, PNX), dtype=bool)
        if c.get("copper", "thin") == "thin":
            # thin grounded copper film directly under the 1-node PTFE layer (paper: 0.01 mm film);
            # the rest of the plate is an insulating support (eps = eps_p)
            dir_[p + self.ytop - 2, p: p + self.Lp] = True
        else:
            dir_[p: p + self.ytop - 1, p: p + self.Lp] = True  # legacy: grounded copper block
        dir_[0, :] = dir_[-1, :] = True              # grounded open boundary
        dir_[:, 0] = dir_[:, -1] = True
        self.dirichlet = dir_
        self.plate_P = np.zeros((PNY, PNX), dtype=bool)
        self.plate_P[p: p + self.ytop, p: p + self.Lp] = True
        self.eps_p, self.eps_w, self.eps_a = c["eps_p"], c["eps_w"], 1.0
        self.sigma_s = np.zeros((PNY, PNX))          # PTFE surface charge (as volume source)
        self.sigma_s[p + self.ytop - 1, p + 1: p + self.Lp - 1] = c["sigma_ptfe"]
        self._lu = None                              # cached factorisation (constant eps)
        self.q_ion = 0.0
        self.psi_el = np.zeros((ny, nx))
        self.Ex = np.zeros((ny, nx))
        self.Ey = np.zeros((ny, nx))

    # -------------------------------------------------------------------------
    def _init_droplet(self):
        c = self.cfg
        yy, xx = np.mgrid[0: self.ny, 0: self.nx]
        self.xc = c["x0"]
        self.yc = self.ytop + c["gap"] + self.R
        r = np.hypot(xx - self.xc, yy - self.yc)
        w = 0.5 * (1 - np.tanh((r - self.R) / 2.0))
        rho = c["rho_v"] + (c["rho_l"] - c["rho_v"]) * w
        rho[self.solid] = c["rho_v"]
        for k in range(9):
            self.f[k] = W9[k] * rho
        self.rho[:] = rho

    def equilibrate(self, n=1500):
        """let the diffuse interface form with no gravity.
        The walls are made neutral meanwhile, otherwise a droplet that starts only 0.55 mm above the
        PTFE (Table S1, H) is pulled onto the surface before it is released and never falls."""
        Gw_save = self.Gw_map.copy()
        self.Gw_map[:] = 0.0
        hyst, geo = self._hyst_on, self._geo_on
        self._hyst_on = self._geo_on = False     # no wetting model while the wall is neutral
        # start-up sound waves would keep the low-viscosity air moving at ~0.15 m/s: every 25 steps
        # the populations are reset to rest (same densities, zero velocity), which damps them.
        # (A higher relaxation time cannot be used for this: it shifts the liquid/gas densities.)
        for it in range(n):
            self._flow_step(0.0, 0.0)
            if it % 25 == 24 and it < n - 100:
                k_macros(self.f, self.rho, self.jx, self.jy, self.psi, self.solid)
                for k in range(9):
                    self.f[k] = W9[k] * self.rho
        self.Gw_map[:] = Gw_save
        self._hyst_on, self._geo_on = hyst, geo

    def kick(self, v_phys):
        """give the droplet an initial velocity (vertical in the lab frame)"""
        c = self.cfg
        v = v_phys * self.v_scale
        vx, vy = v * np.sin(self.alpha), -v * np.cos(self.alpha)
        k_macros(self.f, self.rho, self.jx, self.jy, self.psi, self.solid)
        mask = (self.rho > 0.5 * (c["rho_l"] + c["rho_v"])) & (~self.solid)
        usq = vx * vx + vy * vy
        for k in range(9):
            eu = EX9[k] * vx + EY9[k] * vy
            feq = W9[k] * self.rho * (1 + 3 * eu + 4.5 * eu ** 2 - 1.5 * usq)
            self.f[k][mask] = feq[mask]

    def _flow_step(self, gx, gy):
        k_macros(self.f, self.rho, self.jx, self.jy, self.psi, self.solid)
        c = self.cfg
        if self._hyst_on:
            frac = (self.rho[self.ytop, : self.Lp] - c["rho_v"]) / (c["rho_l"] - c["rho_v"])
            self.wet_mem |= frac > c.get("wet_on", 0.5)
            self.wet_mem &= ~(frac < c.get("wet_off", 0.1))
            self.Gw_map[self.ytop - 1, : self.Lp] = np.where(self.wet_mem, c["Gw_wet"], c["Gw"])
        if self._geo_on:
            self._geometric_wall()
        if c.get("collision", "bgk") == "mrt":
            s_nu = 1.0 / c["tau"]
            S = np.array([0.0, c.get("s_e", 1.0), c.get("s_eps", 1.0), 0.0, c.get("s_q", 1.1),
                          0.0, c.get("s_q", 1.1), s_nu, s_nu])
            k_collide_mrt(self.f, self.ftmp, self.rho, self.jx, self.jy, self.ux, self.uy,
                          self.psi, self.solid, c["G"], self.Gw_map, gx, gy, S)
        else:
            k_collide(self.f, self.ftmp, self.rho, self.jx, self.jy, self.ux, self.uy,
                      self.psi, self.solid, c["G"], self.Gw_map, gx, gy, c["tau"])
        k_stream9(self.f, self.ftmp, self.solid, self.ytop - 1, self.cfg.get("r_bb", 1.0))

    def _geometric_wall(self):
        """virtual wall density for the PTFE layer (geometric wetting + contact-angle window)"""
        c = self.cfg
        L, y1 = self.Lp, self.ytop
        r1 = self.rho[y1, :L]
        r2 = self.rho[y1 + 1, :L]
        dx = np.abs(np.gradient(r1))                    # |d rho / dx| along the wall, first fluid layer
        dn = r1 - r2                                     # minus the normal gradient (into the fluid)
        th_c = np.pi / 2 - np.arctan2(dn, dx + 1e-12)   # current local contact angle
        th = np.clip(th_c, np.deg2rad(c["theta_r"]), np.deg2rad(c["theta_a"]))
        rw = r1 + np.tan(np.pi / 2 - th) * dx
        rw = np.clip(rw, 0.8 * c["rho_v"], 1.05 * c["rho_l"])
        self.Gw_map[y1 - 1, :L] = c["G"] * (1.0 - np.exp(-rw))   # force = -G psi(x) psi_w w_k e_k

    # -------------------------------------------------------------------------
    def water_fraction(self):
        c = self.cfg
        m = (self.rho - c["rho_v"]) / (c["rho_l"] - c["rho_v"])
        m = np.clip(m, 0, 1)
        m[self.solid] = 0
        return m

    def init_ions(self):
        c = self.cfg
        m = self.water_fraction()
        self.cion = m.copy()
        for k in range(5):
            self.g[k] = W5[k] * self.cion
        # charge per unit concentration chosen so droplet carries a given fraction
        # of the total PTFE charge
        q_ptfe_tot = abs(c["sigma_ptfe"]) * (self.Lp - 2)
        self.q_ion = c["drop_charge_frac"] * q_ptfe_tot / self.cion.sum()
        # diffusion (D = cs^2 (tau-0.5)), zero-ish in air as in the paper
        D = c["D_air"] + (c["D_w"] - c["D_air"]) * m
        self.tauc = 0.5 + 3.0 * D
        self.mu = c["mig_ratio"] * c["v_ref"] / abs(c["sigma_ptfe"])

    def poisson(self):
        p = self.pad
        const_eps = (self.eps_w == self.eps_a)       # then the matrix never changes
        if const_eps and self._lu is not None:
            lu, free = self._lu
        else:
            eps = np.full(self.dirichlet.shape, self.eps_a)
            m = self.water_fraction()
            eps[p: p + self.ny, p: p + self.nx] = self.eps_a + (self.eps_w - self.eps_a) * m
            eps[self.plate_P] = self.eps_p
            lu, free = solve_poisson(eps, None, self.dirichlet)
            if const_eps:
                self._lu = (lu, free)
        rho_q = self.sigma_s.copy()
        rho_q[p: p + self.ny, p: p + self.nx] += self.q_ion * self.cion
        psiP = solve_poisson_lu(lu, free, rho_q)
        ExP = np.zeros_like(psiP)
        EyP = np.zeros_like(psiP)
        ExP[:, 1:-1] = -(psiP[:, 2:] - psiP[:, :-2]) / 2
        EyP[1:-1, :] = -(psiP[2:, :] - psiP[:-2, :]) / 2
        sl = (slice(p, p + self.ny), slice(p, p + self.nx))
        self.psi_el, self.Ex, self.Ey = psiP[sl], ExP[sl], EyP[sl]
        return self.electrode_charge()

    def electrode_charge(self):
        # Gauss law on the grounded copper surface: Q = -eps_p * sum( d psi / dn )
        return -self.eps_p * self.psi_el[self.ytop - 1, 1: self.Lp - 1].sum()

    def _ion_step(self):
        m = self.water_fraction()
        vmig = np.clip(self.mu * self.Ex, -0.1, 0.1), np.clip(self.mu * self.Ey, -0.1, 0.1)
        if self.cfg.get("ion_confinement", "diffuse_domain") == "diffuse_domain":
            # ions live in the water only (the paper's no-flux condition at the droplet surface).
            # Diffuse-domain form: d c/dt + div(c u) = div(D m grad(c/m)) + migration, i.e. the
            # usual diffusion plus a drift D grad(m)/m into the water. Its equilibrium is c ~ m:
            # uniform inside the water, zero in the air, with no artificial pile-up anywhere.
            D = (self.tauc - 0.5) / 3.0
            gmx = np.zeros_like(m)
            gmy = np.zeros_like(m)
            gmx[:, 1:-1] = 0.5 * (m[:, 2:] - m[:, :-2])
            gmy[1:-1, :] = 0.5 * (m[2:, :] - m[:-2, :])
            inv = 1.0 / np.maximum(m, 0.02)
            # in the interface band (m < 0.9) the pull back into the water is made stronger, so ions
            # keep up with a receding contact line instead of being left on the plate; the drift
            # stays of the form K(m) grad(m)/m, whose equilibrium concentration rises monotonically
            # with m, so it cannot create a pile-up inside the water
            band = np.clip((0.95 - m) / 0.05, 0.0, 1.0)
            K = D + self.cfg.get("conf_band", 0.0) * band
            vx = self.ux + m * vmig[0] + K * gmx * inv
            vy = self.uy + m * vmig[1] + K * gmy * inv
            vx = np.clip(vx, -0.25, 0.25)
            vy = np.clip(vy, -0.25, 0.25)
            k_ion_collide(self.g, self.gtmp, self.cion, vx, vy, self.tauc, self.solid)
            k_ion_stream(self.g, self.gtmp, self.solid)
            if self.cfg.get("ion_limiter", False):
                tot = self.g.sum()
                np.maximum(self.g, 0.0, out=self.g)
                new_tot = self.g.sum()
                if new_tot > 0:
                    self.g *= tot / new_tot
            return
        # legacy model (ion_confinement = "legacy"): advection with m*u plus a drift conf_k*grad(m).
        # NOTE: with conf_k >> D its equilibrium is c ~ exp(conf_k/D * m), which piles the ions up
        # wherever the water fraction is slightly higher (it was replaced by the form above).
        vx = m * (self.ux + vmig[0])
        vy = m * (self.uy + vmig[1])
        kc = self.cfg.get("conf_k", 0.0)
        if kc > 0:
            gmx = np.zeros_like(m)
            gmy = np.zeros_like(m)
            gmx[:, 1:-1] = 0.5 * (m[:, 2:] - m[:, :-2])
            gmy[1:-1, :] = 0.5 * (m[2:, :] - m[:-2, :])
            # act only in the interface band: inside the water the fraction varies by ~1 %
            # (slight density layering at the wall), and a drift there would sweep all ions into
            # the wall layer
            w_if = np.clip((0.95 - m) / 0.1, 0.0, 1.0)
            vx = vx + kc * w_if * gmx
            vy = vy + kc * w_if * gmy
        vx = np.clip(vx, -0.25, 0.25)
        vy = np.clip(vy, -0.25, 0.25)
        k_ion_collide(self.g, self.gtmp, self.cion, vx, vy, self.tauc, self.solid)
        k_ion_stream(self.g, self.gtmp, self.solid)
        if self.cfg.get("ion_limiter", False):
            # positivity limiter: remove negative ion populations (numerical undershoot at high
            # cell Peclet number) and rescale so the total droplet charge is conserved exactly
            tot = self.g.sum()
            np.maximum(self.g, 0.0, out=self.g)
            new_tot = self.g.sum()
            if new_tot > 0:
                self.g *= tot / new_tot

    # -------------------------------------------------------------------------
    def contact_length(self):
        c = self.cfg
        row = self.rho[self.ytop, : self.Lp]
        return int((row > 0.5 * (c["rho_l"] + c["rho_v"])).sum())

    def com_x(self):
        c = self.cfg
        m = self.water_fraction()
        return (m * np.arange(self.nx)[None, :]).sum() / max(m.sum(), 1e-9)

    def com_y(self):
        m = self.water_fraction()
        return (m * np.arange(self.ny)[:, None]).sum() / max(m.sum(), 1e-9)

    def run(self, v_phys, max_steps=30000, pois_every=50, snap_every=250, verbose=True):
        c = self.cfg
        self.equilibrate(c["n_equil"])
        self.init_ions()
        Q_ref = None
        # reference: PTFE-only electrode charge
        cion_save = self.cion.copy()
        self.cion[:] = 0
        Q_ref = self.poisson()
        self.cion[:] = cion_save
        self.kick(v_phys)
        gx = self.g_lat * np.sin(self.alpha)
        gy = -self.g_lat * np.cos(self.alpha)
        hist = dict(t=[], Q=[], contact=[], xcom=[], ycom=[])
        snaps = []
        t0 = time.time()
        Q = self.poisson()
        had_contact, zero_count = False, 0
        for step in range(max_steps + 1):
            if step % pois_every == 0:
                Q = self.poisson()
                hist["t"].append(step * self.t_scale)
                hist["Q"].append(Q / Q_ref)
                hist["contact"].append(self.contact_length())
                hist["xcom"].append(self.com_x())
                hist["ycom"].append(self.com_y())
            # finer frames in the first ~40 ms (phase I = contact, phase II = early impact)
            if step % snap_every == 0 or (step < 2000 and step % 100 == 0):
                f32 = np.float32
                snaps.append(dict(step=step, t=step * self.t_scale,
                                  rho=self.rho.astype(f32), ux=self.ux.astype(f32), uy=self.uy.astype(f32),
                                  c=self.cion.astype(f32), contact=self.contact_length()))
            self._flow_step(gx, gy)
            self._ion_step()
            if step % 2000 == 0 and verbose:
                print(f"  step {step:6d}  t={step*self.t_scale*1e3:7.2f} ms  "
                      f"x_com={self.com_x():6.1f}  contact={self.contact_length():3d}  "
                      f"Q/Qref={Q/Q_ref:.4f}  [{time.time()-t0:5.1f}s]", flush=True)
            # stop shortly after the droplet has left the plate (no contact for ~2800 steps)
            cl = self.contact_length()
            had_contact = had_contact or cl > 0
            zero_count = zero_count + 1 if (had_contact and cl == 0) else 0
            if zero_count > 2800:
                break
        return hist, snaps, Q_ref


DEFAULT_CFG = dict(
    # geometry: 2D, droplet radius 2 mm (Table S1, R), 30 deg incline = paper's base case (Fig. 3c
    # peaks match the alpha = 30 deg curve of Fig. 3e), plate + domain ~45-60 mm. 1 node = 0.1 mm.
    # gap = 2.2 mm release height: reproduces the paper's landing-peak times (Fig. 3c); Table S1
    # lists H = 0.55 mm, but with that height the landing peaks come 10-15 ms too early.
    # x0 (start position along the plate, not given in the paper) sets the release time.
    nx=580, ny=230, R=20, Lp=420, ytop=100, x0=140, gap=22, alpha_deg=30.0,
    # Shan-Chen fluid with MRT collision + exact-difference forcing, calibrated with calib_mrt.py /
    # calib_gw.py: G = -5 gives rho_l = 1.952, rho_v = 0.158, sigma = 0.056 (Laplace test);
    # Gw = -2.33 gives a static contact angle of ~105 deg (Table S1). tau = 0.55 is ~6x less viscous
    # than the old BGK model (BGK was unstable below tau = 0.6); tau = 0.53 here.
    collision="mrt", s_e=1.0, s_eps=1.0, s_q=1.1,
    G=-5.0, tau=0.53, rho_l=1.952, rho_v=0.158, sigma_lat=0.056, Gw=-2.33,
    n_equil=1500,
    # PTFE wetting: geometric condition with a contact-angle window (see _geometric_wall):
    # advancing angle 105 deg (Table S1); inside [theta_r, theta_a] the wall exerts no capillary
    # force, so the contact line is carried by the slip flow and the droplet keeps the stretched
    # shape it gets at impact, like the paper's phase-field droplet (Fig. 3a, phases II-III).
    # wall_model="force" uses the constant wall strength Gw instead (static angle 105 deg).
    wall_model="geometric", theta_a=105.0, theta_r=20.0,
    r_bb=0.02,       # PTFE wall = Navier (partial) slip, as in the paper (Sec. 2.7): 2 % bounce-back
                     # + 98 % specular reflection. 1 = no-slip, 0 = free slip. A little friction
                     # makes the near-wall layer slower than the top (paper's arrows), so the
                     # front thickens and the rear thins; more friction stalls the droplet.
    # ions / electrostatics (arbitrary but consistent units, eps0 = 1)
    sigma_ptfe=-1.0, eps_p=2.0, eps_w=1.0, drop_charge_frac=0.029, Gw_far=0.0,
    D_w=0.02, D_air=0.02, v_ref=0.02,
    mig_ratio=1.0,   # electric migration toward the charged PTFE (double layer at the contact area,
                     # paper Sec. 3.2). Fitted: 1 gives the paper's release peak at v0 = 0; much
                     # stronger values pack the ions against the PTFE and the release becomes a spike.
    ion_confinement="diffuse_domain",   # ions stay in the water (no-flux at the droplet surface)
    conf_band=0.06,  # extra pull back into the water inside the interface band, so ions follow a
                     # receding contact line instead of being left on the plate
    conf_k=0.5,      # only used by ion_confinement="legacy"
    pad=300,         # extra air nodes around the Poisson domain (grounded boundary farther away)
    ion_limiter=True,   # keep ion concentration non-negative (charge-conserving)
    copper="thin",   # "thin" = 1-node copper film (like the paper), "block" = old behaviour
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
                    help="initial droplet velocities in m/s")
    ap.add_argument("--cfg", type=str, default=None, help="json file overriding DEFAULT_CFG")
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--out", type=str, default="fig3_results.npz")
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG)
    if a.cfg:
        cfg.update(json.load(open(a.cfg)))
    import pickle
    results = {}
    for v in a.v:
        print(f"=== v0 = {v} m/s ===", flush=True)
        sim = DropletSim(cfg)
        hist, snaps, Qref = sim.run(v, max_steps=a.max_steps)
        results[v] = dict(hist=hist, snaps=snaps, Qref=Qref, t_scale=sim.t_scale)
        pickle.dump(dict(cfg=cfg, results=results), open(a.out.replace(".npz", ".pkl"), "wb"))  # save after every case
    print("saved", a.out.replace(".npz", ".pkl"))
