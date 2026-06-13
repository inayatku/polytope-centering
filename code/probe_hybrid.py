"""Why does the hybrid land at CN-like quality in higher dimensions?
Hypothesis A: CN's fixed point nearly satisfies P's Delta-stopping rule,
so the P phase quits immediately (stopping artifact).
Hypothesis B: P genuinely converges to a different, less central
balanced-chord point when started from CN's limit (multiple attractors).
Decide by running the P phase to much tighter tolerance."""

import numpy as np

from centers import cn_center, p_center
from polytope import random_simplex_like

for rep in range(5):
    rng = np.random.default_rng(1000 * 200 + rep)
    P, x0 = random_simplex_like(200, 20, rng)
    rp = p_center(P, x0, 1e-3, 5000)
    rc = cn_center(P, x0, 1e-2, 5000)          # hybrid's CN phase (10*eps)
    rh3 = p_center(P, rc.x, 1e-3, 5000)        # hybrid's P phase
    rh6 = p_center(P, rc.x, 1e-6, 20000)       # tight P phase
    rp6 = p_center(P, x0, 1e-6, 20000)         # tight P from scratch
    print(f"rep {rep}: P(1e-3) {P.rel_centrality(rp.x):.3f} "
          f"| CNphase {P.rel_centrality(rc.x):.3f} "
          f"| P-after-CN(1e-3) {P.rel_centrality(rh3.x):.3f} "
          f"({rh3.sweeps} sw) "
          f"| P-after-CN(1e-6) {P.rel_centrality(rh6.x):.3f} "
          f"({rh6.sweeps} sw) "
          f"| P-scratch(1e-6) {P.rel_centrality(rp6.x):.3f} "
          f"| dist(limits) {np.linalg.norm(rh6.x - rp6.x):.4f}")
