"""Generate paper figures from ../results CSVs into ../figures (PDF + PNG)."""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")
os.makedirs(FIGURES, exist_ok=True)

# Figures are built ~13.5in wide and included at \textwidth (~6.4in), i.e.
# ~47% scale. Fonts are set so in-print text lands at ~8pt. Palette is
# Okabe-Ito (colorblind-safe); linestyles differ so curves survive grayscale.
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
})

STYLE = {
    "P": dict(color="#E69F00", marker="o", ls="-",
              label="Jacobi (P-center)"),
    "CN": dict(color="#009E73", marker="D", ls="--",
               label="recursive (CN-center)"),
    "HYB": dict(color="#D55E00", marker="s", ls="-.",
                label="hybrid CN$\\to$P"),
    "CN-rand": dict(color="#0072B2", marker="^", ls=":",
                    label="CN random (tail-avg)"),
    "CN-greedy": dict(color="#CC79A7", marker="v", ls="-",
                      label="CN greedy (tail-avg)"),
}


def load(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        return header, list(r)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf/.png")


# ---------------------------------------------------------------- figure 1
def fig1():
    _, traj = load("e1_trajectories.csv")
    _, geo = load("e1_geometry.csv")
    T = defaultdict(list)
    for tag, sweep, chords, x, y in traj:
        T[tag].append((float(x), float(y)))
    G = defaultdict(dict)
    verts = defaultdict(list)
    for poly, kind, x, y in geo:
        if kind == "vertex":
            verts[poly].append((float(x), float(y)))
        else:
            G[poly][kind] = (float(x), float(y))

    panels = [("wedge", "wedge_narrow", "(a) wedge, narrow-corner start"),
              ("wedge", "wedge_wide", "(b) wedge, wide-corner start"),
              ("rand", "rand", "(c) random polytope ($m{=}25$, $n{=}2$)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (poly, prefix, title) in zip(axes, panels):
        V = np.array(verts[poly] + [verts[poly][0]])
        ax.fill(V[:, 0], V[:, 1], color="#dce9f5", zorder=0)
        ax.plot(V[:, 0], V[:, 1], "k-", lw=1)
        for meth in ("P", "CN", "HYB"):
            pts = np.array(T[f"{prefix}_{meth}"])
            st = STYLE[meth]
            ax.plot(pts[:, 0], pts[:, 1], st["ls"], color=st["color"],
                    lw=1.5, alpha=0.7)
            ax.plot(pts[:, 0], pts[:, 1], st["marker"], color=st["color"],
                    ms=5.5, label=st["label"])
        ax.plot(*G[poly]["chebyshev"], "k*", ms=19, label="Chebyshev")
        ax.plot(*G[poly]["analytic"], "ks", mfc="none", ms=12,
                label="analytic")
        ax.plot(*G[poly]["centroid"], "k^", mfc="none", ms=12,
                label="centroid")
        ax.set_title(title, fontsize=15)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.08),
               frameon=False)
    savefig(fig, "fig1_trajectories")


# ---------------------------------------------------------------- figure 2
def fig2():
    _, rows = load("e2_convergence.csv")
    data = defaultdict(list)
    for inst, meth, sweep, chords, relc in rows:
        data[(inst, meth)].append((int(chords), float(relc)))
    panels = [("wedge", "(a) wedge, narrow-corner start"),
              ("rand100x10", "(b) random polytope ($m{=}100$, $n{=}10$)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (inst, title) in zip(axes, panels):
        for meth, st in STYLE.items():
            pts = np.array(sorted(data[(inst, meth)]))
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], st["ls"], color=st["color"],
                    marker=st["marker"], ms=5,
                    markevery=max(1, len(pts) // 25),
                    label=st["label"], lw=1.8)
        ax.set_xlabel("chord evaluations")
        ax.set_ylabel("relative centrality  $C(x)/r_{cheb}$")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig2_convergence")


# ---------------------------------------------------------------- figure 3
def fig3():
    _, rows = load("e3_omega.csv")
    agg = defaultdict(list)
    for geom, meth, omega, sweeps, relc, conv in rows:
        agg[(geom, meth, float(omega))].append((int(sweeps), float(relc)))
    geoms = [("wedge", "(a) wedge"),
             ("rand100x10", "(b) random $100{\\times}10$"),
             ("aniso50x5", "(c) anisotropic $50{\\times}5$, $\\kappa{=}0.1$")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=False)
    for ax, (geom, title) in zip(axes, geoms):
        for meth in ("P", "CN"):
            omegas = sorted({o for (g, m, o) in agg if g == geom and m == meth})
            mean_sw = [np.mean([s for s, _ in agg[(geom, meth, o)]])
                       for o in omegas]
            st = STYLE[meth]
            ax.plot(omegas, mean_sw, st["ls"], color=st["color"],
                    marker=st["marker"], ms=7, lw=1.8, label=st["label"])
        ax.set_xlabel("relaxation $\\omega$")
        ax.set_ylabel("sweeps to $\\Delta < 10^{-3}$")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    savefig(fig, "fig3_omega")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
