"""Probe: does the CN advantage reappear on thin / narrow-corner polytopes
(the regime of Figures 1 and 6 of the original paper)? Also compare the
paper's recursion (averaging running points) against the natural
alternative (running mean of midpoints) via the block parameter.

Thin triangle: apex at origin with opening 'kappa', length 1:
    -y <= 0,   y - kappa*x <= 0 (slanted lid),   x <= 1
Start just inside the apex -> exactly the 'narrow corner' regime.
"""

import numpy as np

from centers import cn_center, p_center
from polytope import Polytope, random_simplex_like

EPS = 1e-3


def thin_triangle(kappa):
    A = np.array([[0.0, -1.0], [-kappa, 1.0], [1.0, 0.0]])
    b = np.array([0.0, 0.0, 1.0])
    P = Polytope(A, b)
    x0 = np.array([0.02, 0.01 * kappa])  # near the sharp apex
    assert P.is_interior(x0)
    return P, x0


print("kappa   P sweeps  relC  |  CN sweeps  relC  |  CN-greedy sweeps relC")
for kappa in [1.0, 0.3, 0.1, 0.03]:
    P, x0 = thin_triangle(kappa)
    rp = p_center(P, x0, eps=EPS, max_sweeps=100000)
    rc = cn_center(P, x0, eps=EPS, max_sweeps=100000)
    rg = cn_center(P, x0, eps=EPS, max_sweeps=100000, ordering="greedy",
                   rng=np.random.default_rng(0))
    print(f"{kappa:5.2f}  {rp.sweeps:7d}  {P.rel_centrality(rp.x):5.3f} | "
          f"{rc.sweeps:8d}  {P.rel_centrality(rc.x):5.3f} | "
          f"{rg.sweeps:8d}  {P.rel_centrality(rg.x):5.3f}")

print("\nAnisotropic random polytopes: scale last coordinate by kappa")
print("kappa   P sweeps  relC  |  CN sweeps  relC")
for kappa in [1.0, 0.1, 0.01]:
    sp, sc, qp, qc = [], [], [], []
    for rep in range(10):
        rng = np.random.default_rng(300 + rep)
        P0, x0 = random_simplex_like(50, 5, rng)
        D = np.eye(5)
        D[-1, -1] = 1.0 / kappa     # x_new = D^{-1} x  => A_new = A @ D
        P = Polytope(P0.A @ np.diag([1, 1, 1, 1, kappa]), P0.b)
        x0 = x0 / np.array([1, 1, 1, 1, 1 / kappa])
        # interior check: A_new x0_new = A (D x0_new) = A x0  -> same slack
        rp = p_center(P, x0, eps=EPS, max_sweeps=100000)
        rc = cn_center(P, x0, eps=EPS, max_sweeps=100000)
        sp.append(rp.sweeps); sc.append(rc.sweeps)
        qp.append(P.rel_centrality(rp.x)); qc.append(P.rel_centrality(rc.x))
    print(f"{kappa:5.2f}  {np.mean(sp):8.1f}  {np.mean(qp):5.3f} | "
          f"{np.mean(sc):9.1f}  {np.mean(qc):5.3f}")

print("\nRunning-point recursion (paper) vs running-mean-of-midpoints")
print("(block=1 paper CN vs sequential mean variant) on thin triangle k=0.1")
P, x0 = thin_triangle(0.1)


def cn_midpoint_mean(P, x0, eps, max_sweeps=100000):
    """CN with x_i = ((i-1) x_{i-1} + mid_i)/i  (online mean of midpoints,
    chords still shot from the latest point)."""
    x = x0.copy()
    for k in range(1, max_sweeps + 1):
        x_prev = x.copy()
        xi = x.copy()
        acc = np.zeros(P.n)
        for i in range(P.m):
            _, _, mid = P.chord(xi, i)
            acc += mid
            xi = acc / (i + 1)
        x = xi
        if np.max(np.abs(x - x_prev)) < eps:
            return x, k
    return x, max_sweeps


x_mm, k_mm = cn_midpoint_mean(P, x0, EPS)
rc = cn_center(P, x0, eps=EPS, max_sweeps=100000)
print(f"paper CN: {rc.sweeps} sweeps, relC={P.rel_centrality(rc.x):.3f}")
print(f"mid-mean: {k_mm} sweeps, relC={P.rel_centrality(x_mm):.3f}")
