"""Pilot run: calibrate omega, check sweep costs and sweeps-to-convergence
so the full experiment grid finishes in reasonable time."""

import time

import numpy as np

from centers import cn_center, p_center
from polytope import random_simplex_like

EPS = 1e-3  # same tolerance the original paper used

print("=== sweep cost at large size (m=1000+50, n=50) ===")
rng = np.random.default_rng(0)
P, x0 = random_simplex_like(1000, 50, rng)
t = time.perf_counter()
r = cn_center(P, x0, eps=0, max_sweeps=3)
per_sweep = (time.perf_counter() - t) / 3
print(f"CN sweep: {per_sweep*1000:.1f} ms  (m={P.m})")

print("\n=== sweeps to eps=1e-3 at (m, n) sizes, 5 reps ===")
for (m, n) in [(25, 2), (100, 10), (500, 20)]:
    sp, sc = [], []
    for rep in range(5):
        rng = np.random.default_rng(100 + rep)
        P, x0 = random_simplex_like(m, n, rng)
        rp = p_center(P, x0, eps=EPS, max_sweeps=5000)
        rc = cn_center(P, x0, eps=EPS, max_sweeps=5000)
        sp.append(rp.sweeps)
        sc.append(rc.sweeps)
    print(f"({m:4d},{n:2d})  P: {np.mean(sp):7.1f} sweeps   CN: {np.mean(sc):6.1f} sweeps")

print("\n=== omega scan for CN-SOR, (m=100, n=10), 10 reps ===")
print("omega   sweeps   relC_final   interior_ok")
for omega in [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9]:
    sw, q, ok = [], [], True
    for rep in range(10):
        rng = np.random.default_rng(200 + rep)
        P, x0 = random_simplex_like(100, 10, rng)
        r = cn_center(P, x0, eps=EPS, max_sweeps=5000, omega=omega)
        sw.append(r.sweeps)
        q.append(P.rel_centrality(r.x))
        ok = ok and P.is_interior(r.x)
    print(f"{omega:.1f}   {np.mean(sw):7.1f}   {np.mean(q):8.3f}      {ok}")

print("\n=== omega scan for P-SOR, (m=100, n=10), 10 reps ===")
for omega in [1.0, 1.5, 1.8, 1.9]:
    sw, q = [], []
    for rep in range(10):
        rng = np.random.default_rng(200 + rep)
        P, x0 = random_simplex_like(100, 10, rng)
        r = p_center(P, x0, eps=EPS, max_sweeps=5000, omega=omega)
        sw.append(r.sweeps)
        q.append(P.rel_centrality(r.x))
    print(f"{omega:.1f}   {np.mean(sw):7.1f}   {np.mean(q):8.3f}")
