"""Reconstruct (approximately) Polytope 1 / Figure 6 of the original paper:
an elongated wedge with a sharp corner at the right, trajectory starting
near the sharp corner. The paper reports P: 56 sweeps vs CN: 11 sweeps.
If the faithful implementation reproduces a gap of this order on this
geometry, the implementation is validated and the 'geometry-dependent
advantage' story is confirmed.
"""

import numpy as np

from centers import cn_center, p_center
from polytope import Polytope


def wedge_from_vertices(V):
    """Polytope from counterclockwise 2-D vertices."""
    V = np.asarray(V, float)
    k = len(V)
    A, b = [], []
    for i in range(k):
        p, q = V[i], V[(i + 1) % k]
        e = q - p
        n_out = np.array([e[1], -e[0]])  # outward for CCW ordering
        A.append(n_out)
        b.append(n_out @ p)
    return Polytope(np.array(A), np.array(b))


# Vertices read off Figure 1 of the original paper (approximate)
V = [(0.2, 2.8), (4.4, 0.75), (0.55, 6.15)]
P = wedge_from_vertices(V)
x0 = np.array([3.5, 1.5])
print("interior:", P.is_interior(x0), " m =", P.m)

for eps in [1e-3, 1e-4]:
    rp = p_center(P, x0, eps=eps, max_sweeps=100000)
    rc = cn_center(P, x0, eps=eps, max_sweeps=100000)
    print(f"eps={eps:g}:  P {rp.sweeps:5d} sweeps -> x={rp.x.round(3)} "
          f"relC={P.rel_centrality(rp.x):.3f} | "
          f"CN {rc.sweeps:4d} sweeps -> x={rc.x.round(3)} "
          f"relC={P.rel_centrality(rc.x):.3f}")

print("\nWide-corner start (paper Figure 7):")
x0w = np.array([0.7, 3.0])
rp = p_center(P, x0w, eps=1e-3, max_sweeps=100000)
rc = cn_center(P, x0w, eps=1e-3, max_sweeps=100000)
print(f"P {rp.sweeps} sweeps relC={P.rel_centrality(rp.x):.3f} | "
      f"CN {rc.sweeps} sweeps relC={P.rel_centrality(rc.x):.3f}")

print("\nReference centers:")
print("  Chebyshev:", P.chebyshev()[0].round(3), " r =", round(P.chebyshev()[1], 4))
print("  analytic :", P.analytic_center(x0).round(3))
print("  centroid :", P.centroid_2d().round(3))
print("  paper reports P-center (1.4453, 3.4772), CN-center (1.3980, 3.6719)")
