#!/usr/bin/env python3
"""3D scientific plot of the non-attainment example (original slide 49).

    minimize  x1^2 + 2(1-x2)^2 + |x1|_0 + 1/2 |x2|_0
    subject to |x1|_0 >= |x2|_0,   -1 <= x1, x2 <= 1.

On the feasible region (x1 != 0) the objective is the paraboloid
    f = x1^2 + 2(1-x2)^2 + 1.5,
whose valley floor lies along the slit x1 = 0.  That slit is infeasible
(|x1|_0 = 0 < |x2|_0 = 1), so it is cut out of the surface.  The infimum 1.5
is approached along (eps, 1) as eps -> 0 but never attained: its limit (0,1)
falls in the removed slit.  Regenerate with:  python3 make_nonattain3d.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- project supplement palette -------------------------------------------
accent = (90 / 255, 119 / 255, 126 / 255)   # SupplementAccent
tint = (241 / 255, 246 / 255, 246 / 255)    # SupplementTint
dark = (0.219, 0.259, 0.349)                # MediumBlack
blue = (0.05, 0.15, 0.35)                   # DarkBlue
teal = LinearSegmentedColormap.from_list(
    "supp_teal", [tint, accent, (0.12, 0.21, 0.24)])

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
})

# --- feasible objective surface (x1 != 0) ---------------------------------
n = 220
xs = np.linspace(-1, 1, n)
ys = np.linspace(-1, 1, n)
X, Y = np.meshgrid(xs, ys)
Z = X**2 + 2 * (1 - Y)**2 + 1.5
delta = 0.06                       # width of the excised infeasible slit
Z[np.abs(X) < delta] = np.nan      # cut the slit x1 = 0 out of the surface

fig = plt.figure(figsize=(6.6, 4.7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap=teal, vmin=1.5, vmax=6.6,
                rcount=150, ccount=150, linewidth=0, antialiased=True,
                alpha=0.9, zorder=1)

# expose the two valley walls flanking the slit
yy = np.linspace(-1, 1, 200)
for sgn in (-1, 1):
    ax.plot(sgn * delta * np.ones_like(yy), yy,
            delta**2 + 2 * (1 - yy)**2 + 1.5,
            color=accent, lw=0.9, alpha=0.85, zorder=3)

# faint contour projection on the floor for a scientific read
zfloor = 0.6
ax.contour(X, Y, Z, levels=8, colors=[accent], linewidths=0.4,
           alpha=0.55, offset=zfloor, zdir="z")

# --- the excised slit drawn on the floor (infeasible set) -----------------
ax.plot([0, 0], [-1, 1], [zfloor, zfloor], color=accent, lw=1.2,
        ls=(0, (4, 3)), zorder=2)
ax.text(0.04, -0.75, zfloor, "infeasible slit", color=accent, fontsize=8,
        zorder=4)

# --- minimizing sequence (eps, 1) -> (0, 1), z = 1.5 + eps^2 --------------
eps = np.array([0.62, 0.42, 0.27, 0.16, 0.08])
zseq = eps**2 + 1.5
ax.plot(eps, np.ones_like(eps), zseq, color=dark, lw=0.9, ls=":", zorder=5)
ax.scatter(eps, np.ones_like(eps), zseq, color=dark, s=22,
           depthshade=False, zorder=6)

# --- unattained infimum (0, 1, 1.5): hollow = not in feasible set ---------
ax.scatter([0], [1], [1.5], facecolors="white", edgecolors=blue,
           s=78, linewidths=1.9, depthshade=False, zorder=7)
ax.plot([0, 0], [1, 1], [zfloor, 1.5], color=blue, lw=0.7, ls=(0, (2, 2)),
        zorder=4)
ax.text(-0.06, 1.05, 2.12, r"$\mathrm{inf}=1.5$ (not attained)",
        color=blue, fontsize=9.5, zorder=8)
ax.text(0.70, 1.0, 1.78, r"$(\varepsilon,1)$", color=dark, fontsize=9)

# --- cosmetics ------------------------------------------------------------
ax.set_xlabel(r"$x_1$", fontsize=11, labelpad=-2)
ax.set_ylabel(r"$x_2$", fontsize=11, labelpad=-2)
ax.set_zlabel(r"$f(x_1,x_2)$", fontsize=11, labelpad=-1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(zfloor, 5.6)
ax.set_zticks([1.5, 3, 4.5])
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.tick_params(labelsize=8, pad=-1)
ax.view_init(elev=26, azim=-62)
ax.set_box_aspect((1, 1, 0.62), zoom=1.05)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor((1, 1, 1, 0))
    pane.pane.set_edgecolor((0.8, 0.8, 0.8, 1))
ax.grid(True, linewidth=0.3, alpha=0.4)

fig.savefig("assets/nonattain3d.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote assets/nonattain3d.pdf")
