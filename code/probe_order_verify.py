"""Verify E6's striking order-dependence: confirm fixed-ordering CN runs
genuinely converge (Delta < 1e-6, no sweep-cap censoring) and that two very
different orderings really converge to far-apart limits."""

import numpy as np

from centers import cn_center
from polytope import random_simplex_like

rng = np.random.default_rng(600)  # instance 0 of E6
P, x0 = random_simplex_like(50, 10, rng)
_, r_cheb = P.chebyshev()
print(f"r_cheb = {r_cheb:.4f}")

limits = []
for p in [0, 7, 13, 29]:
    perm = np.random.default_rng(p).permutation(P.m)
    r = cn_center(P, x0, eps=1e-6, max_sweeps=50000, ordering="given",
                  perm=perm)
    # continue from the limit with the same ordering: should not move
    r2 = cn_center(P, r.x, eps=1e-8, max_sweeps=50000, ordering="given",
                   perm=perm)
    move = np.linalg.norm(r2.x - r.x)
    limits.append(r.x)
    print(f"perm {p:2d}: converged={r.converged} sweeps={r.sweeps} "
          f"relC={P.rel_centrality(r.x):.3f} "
          f"extra-move at 1e-8: {move:.2e}")

L = np.array(limits)
for i in range(len(L)):
    for j in range(i + 1, len(L)):
        d = np.linalg.norm(L[i] - L[j])
        print(f"dist(perm{i},perm{j}) = {d:.4f} = {d / r_cheb:.2f} r_cheb")
