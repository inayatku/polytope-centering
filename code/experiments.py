"""Full experiment suite for the follow-up paper. Writes CSVs to ../results.

E1  2-D trajectories (wedge narrow/wide start, random polytope)   -> e1_*.csv
E2  convergence curves, relC vs chords                            -> e2_*.csv
E3  omega ablation for P-SOR / CN-SOR on three geometries         -> e3_omega.csv
E4  scaling study over (m, n) grid, all methods + baselines       -> e4_scaling.csv
E5  redundancy stress test (duplicated constraints)               -> e5_redundancy.csv
E6  order dependence of the CN limit                              -> e6_order.csv

Run:  python experiments.py [e1 e2 e3 e4 e5 e6]   (default: all)
"""

import csv
import os
import sys
import time

import numpy as np

from centers import cn_center, hybrid_center, p_center, tail_average
from polytope import (Polytope, duplicate_constraints, multiplicity_weights,
                      random_simplex_like)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

EPS = 1e-3          # sweep-level stopping tolerance (as in the original paper)
MAX_SWEEPS = 5000
BUDGET = 30         # sweep budget for ordering variants without Delta-stopping


def save_csv(name, header, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def wedge_polytope():
    """Approximate reconstruction of Polytope 1 / Figure 6 of the original
    paper: elongated wedge with a sharp right corner."""
    V = np.array([(0.2, 2.8), (4.4, 0.75), (0.55, 6.15)])
    k = len(V)
    A, b = [], []
    for i in range(k):
        p, q = V[i], V[(i + 1) % k]
        e = q - p
        n_out = np.array([e[1], -e[0]])
        A.append(n_out)
        b.append(n_out @ p)
    return Polytope(np.array(A), np.array(b))


def aniso_polytope(m, n, kappa, rng):
    """Random simplex-like polytope with the last coordinate scaled by kappa
    (condition-number stress; the polytope becomes a thin slab)."""
    P0, x0 = random_simplex_like(m, n, rng)
    scale = np.ones(n)
    scale[-1] = kappa
    return Polytope(P0.A * scale[None, :], P0.b), x0 / scale


# ---------------------------------------------------------------------------
def e1():
    """Trajectories for the 2-D figure."""
    rows = []

    def record(tag, P, res):
        for k, (c, x) in enumerate(res.history):
            rows.append([tag, k, c, x[0], x[1]])

    W = wedge_polytope()
    for tag, x0 in [("narrow", np.array([3.5, 1.5])),
                    ("wide", np.array([0.7, 3.0]))]:
        record(f"wedge_{tag}_P", W, p_center(W, x0, EPS, MAX_SWEEPS))
        record(f"wedge_{tag}_CN", W, cn_center(W, x0, EPS, MAX_SWEEPS))
        record(f"wedge_{tag}_HYB", W, hybrid_center(W, x0, EPS, MAX_SWEEPS))

    rng = np.random.default_rng(42)
    R, x0 = random_simplex_like(25, 2, rng)
    record("rand_P", R, p_center(R, x0, EPS, MAX_SWEEPS))
    record("rand_CN", R, cn_center(R, x0, EPS, MAX_SWEEPS))
    record("rand_HYB", R, hybrid_center(R, x0, EPS, MAX_SWEEPS))
    save_csv("e1_trajectories.csv", ["tag", "sweep", "chords", "x", "y"], rows)

    # polytope geometry + reference centers for plotting
    geo = []
    for name, P in [("wedge", W), ("rand", R)]:
        for v in P.vertices_2d():
            geo.append([name, "vertex", v[0], v[1]])
        c, r = P.chebyshev()
        geo.append([name, "chebyshev", c[0], c[1]])
        a = P.analytic_center(np.mean(P.vertices_2d(), axis=0))
        geo.append([name, "analytic", a[0], a[1]])
        g = P.centroid_2d()
        geo.append([name, "centroid", g[0], g[1]])
    save_csv("e1_geometry.csv", ["polytope", "kind", "x", "y"], geo)


# ---------------------------------------------------------------------------
def e2():
    """Quality (relC) versus work (chords) curves."""
    rows = []

    def record(inst, tag, P, res, tail=0):
        hist = res.history
        for k, (c, x) in enumerate(hist):
            if tail and k >= tail:
                x = np.mean([h[1] for h in hist[max(0, k - tail):k + 1]], axis=0)
            rows.append([inst, tag, k, c, P.rel_centrality(x)])

    cases = [("wedge", wedge_polytope(), np.array([3.5, 1.5]))]
    rng = np.random.default_rng(7)
    P, x0 = random_simplex_like(100, 10, rng)
    cases.append(("rand100x10", P, x0))
    for inst, P, x0 in cases:
        budget = 120
        record(inst, "P", P, p_center(P, x0, 0.0, budget))
        record(inst, "CN", P, cn_center(P, x0, 0.0, budget))
        record(inst, "HYB", P, hybrid_center(P, x0, EPS, budget))
        record(inst, "CN-rand", P,
               cn_center(P, x0, 0.0, budget, ordering="random",
                         rng=np.random.default_rng(1)), tail=10)
        record(inst, "CN-greedy", P,
               cn_center(P, x0, 0.0, budget, ordering="greedy",
                         rng=np.random.default_rng(1)), tail=10)
    save_csv("e2_convergence.csv",
             ["instance", "method", "sweep", "chords", "relC"], rows)


# ---------------------------------------------------------------------------
def e3():
    """Over-relaxation ablation on three geometries."""
    rows = []
    omegas = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    def run(geom, P, x0, omega):
        rp = p_center(P, x0, EPS, MAX_SWEEPS, omega=omega)
        rc = cn_center(P, x0, EPS, MAX_SWEEPS, omega=omega)
        rows.append([geom, "P", omega, rp.sweeps, P.rel_centrality(rp.x),
                     int(rp.converged)])
        rows.append([geom, "CN", omega, rc.sweeps, P.rel_centrality(rc.x),
                     int(rc.converged)])

    W = wedge_polytope()
    for om in omegas:
        run("wedge", W, np.array([3.5, 1.5]), om)
    for rep in range(10):
        rng = np.random.default_rng(200 + rep)
        P, x0 = random_simplex_like(100, 10, rng)
        for om in omegas:
            run("rand100x10", P, x0, om)
    for rep in range(10):
        rng = np.random.default_rng(300 + rep)
        P, x0 = aniso_polytope(50, 5, 0.1, rng)
        for om in omegas:
            run("aniso50x5", P, x0, om)
    save_csv("e3_omega.csv",
             ["geometry", "method", "omega", "sweeps", "relC", "converged"],
             rows)


# ---------------------------------------------------------------------------
def e4():
    """Scaling study: methods x sizes, work-fair metrics."""
    rows = []
    grid = [(25, 2, 30), (50, 2, 30), (100, 10, 30), (200, 20, 30),
            (500, 20, 30), (1000, 50, 10)]
    for m, n, reps in grid:
        for rep in range(reps):
            rng = np.random.default_rng(1000 * m + rep)
            P, x0 = random_simplex_like(m, n, rng)

            def relc_at(res, sweeps):
                hist = res.history
                k = min(sweeps, len(hist) - 1)
                return P.rel_centrality(hist[k][1])

            t = time.perf_counter()
            c_cheb, r_cheb = P.chebyshev()
            t_cheb = time.perf_counter() - t
            t = time.perf_counter()
            x_ana = P.analytic_center(x0)
            t_ana = time.perf_counter() - t

            res = {
                "P": p_center(P, x0, EPS, MAX_SWEEPS),
                "CN": cn_center(P, x0, EPS, MAX_SWEEPS),
                "HYB": hybrid_center(P, x0, EPS, MAX_SWEEPS),
            }
            for name, r in res.items():
                rows.append([m, n, rep, name, r.sweeps, r.chords, r.seconds,
                             P.rel_centrality(r.x), relc_at(r, 5),
                             relc_at(r, BUDGET), int(r.converged)])
            for name, ordering in [("CN-rand", "random"),
                                   ("CN-greedy", "greedy")]:
                r = cn_center(P, x0, 0.0, BUDGET, ordering=ordering,
                              rng=np.random.default_rng(rep))
                xt = tail_average(r, 10)
                rows.append([m, n, rep, name, r.sweeps, r.chords, r.seconds,
                             P.rel_centrality(xt), relc_at(r, 5),
                             P.rel_centrality(xt), 0])
            rows.append([m, n, rep, "analytic", 0, 0, t_ana,
                         P.rel_centrality(x_ana), np.nan, np.nan, 1])
            rows.append([m, n, rep, "chebyshev", 0, 0, t_cheb,
                         1.0, np.nan, np.nan, 1])
        print(f"  e4: ({m},{n}) done")
    save_csv("e4_scaling.csv",
             ["m", "n", "rep", "method", "sweeps", "chords", "seconds",
              "relC_final", "relC@5", "relC@30", "converged"], rows)


# ---------------------------------------------------------------------------
def e5():
    """Redundancy stress: duplicate half the constraints x2 / x5."""
    rows = []
    for rep in range(30):
        rng = np.random.default_rng(500 + rep)
        P, x0 = random_simplex_like(50, 10, rng)
        _, r_cheb = P.chebyshev()
        base = {
            "P": p_center(P, x0, EPS, MAX_SWEEPS).x,
            "CN": cn_center(P, x0, EPS, MAX_SWEEPS).x,
            "P-w": p_center(P, x0, EPS, MAX_SWEEPS,
                            weights=multiplicity_weights(P)).x,
            "CN-w": cn_center(P, x0, EPS, MAX_SWEEPS,
                              weights=multiplicity_weights(P)).x,
            "analytic": P.analytic_center(x0),
            "chebyshev": P.chebyshev()[0],
        }
        for copies in [2, 5]:
            Pd = duplicate_constraints(P, copies, np.random.default_rng(rep),
                                       frac=0.5)
            dup = {
                "P": p_center(Pd, x0, EPS, MAX_SWEEPS).x,
                "CN": cn_center(Pd, x0, EPS, MAX_SWEEPS).x,
                "P-w": p_center(Pd, x0, EPS, MAX_SWEEPS,
                                weights=multiplicity_weights(Pd)).x,
                "CN-w": cn_center(Pd, x0, EPS, MAX_SWEEPS,
                                  weights=multiplicity_weights(Pd)).x,
                "analytic": Pd.analytic_center(x0),
                "chebyshev": Pd.chebyshev()[0],
            }
            for meth in base:
                drift = np.linalg.norm(dup[meth] - base[meth]) / r_cheb
                rows.append([rep, copies, meth, drift])
    save_csv("e5_redundancy.csv", ["rep", "copies", "method", "drift"], rows)


# ---------------------------------------------------------------------------
def e6():
    """Order dependence: dispersion of the CN limit across permutations.
    High sweep cap (20000) + per-run convergence recording so the dispersion
    cannot be dismissed as a censoring artifact."""
    rows = []
    for inst in range(10):
        rng = np.random.default_rng(600 + inst)
        P, x0 = random_simplex_like(50, 10, rng)
        _, r_cheb = P.chebyshev()
        limits, convs, sws = [], [], []
        for p in range(30):
            perm = np.random.default_rng(p).permutation(P.m)
            r = cn_center(P, x0, eps=1e-6, max_sweeps=20000,
                          ordering="given", perm=perm)
            limits.append(r.x)
            convs.append(int(r.converged))
            sws.append(r.sweeps)
        L = np.array(limits)
        centroid_L = L.mean(axis=0)
        disp = np.linalg.norm(L - centroid_L, axis=1)
        diam = np.max([np.linalg.norm(a - c) for a in L for c in L])
        # tail-averaged randomized run for comparison
        rr = cn_center(P, x0, eps=0.0, max_sweeps=60, ordering="random",
                       rng=np.random.default_rng(inst))
        xt = tail_average(rr, 20)
        rows.append([inst, disp.mean() / r_cheb, diam / r_cheb,
                     P.rel_centrality(centroid_L),
                     np.mean([P.rel_centrality(x) for x in L]),
                     P.rel_centrality(xt),
                     np.mean(convs), np.mean(sws), np.max(sws)])
        print(f"  e6: instance {inst} done "
              f"(converged {np.mean(convs):.0%}, mean sweeps {np.mean(sws):.0f})")
    save_csv("e6_order.csv",
             ["instance", "mean_disp_over_rcheb", "diam_over_rcheb",
              "relC_of_mean", "mean_relC", "relC_rand_tail",
              "converged_frac", "mean_sweeps", "max_sweeps"], rows)


# ---------------------------------------------------------------------------
def sphere_polar(m, n, rng):
    """Second instance family: rows uniform on the unit sphere, b = 1.
    The polytope is the polar of the point set's convex hull; bounded with
    high probability for m >> n (verified by LP, regenerated otherwise).
    Geometry is round rather than simplex-like - a robustness check that the
    E4 conclusions are not an artifact of one generator."""
    from scipy.optimize import linprog
    while True:
        A = rng.normal(size=(m, n))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        b = np.ones(m)
        bounded = True
        for j in range(n):
            for sign in (1.0, -1.0):
                c = np.zeros(n)
                c[j] = -sign
                res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n,
                              method="highs")
                if res.status == 3:  # unbounded
                    bounded = False
                    break
            if not bounded:
                break
        if not bounded:
            continue
        P = Polytope(A, b)
        d = rng.normal(size=n)
        d /= np.linalg.norm(d)
        Ad = A @ d
        t_max = np.min(1.0 / Ad[Ad > 1e-12])
        x0 = 0.9 * t_max * d  # near-boundary start
        assert P.is_interior(x0)
        return P, x0


def e7():
    """Scaling on the second (sphere-polar) family - replication check."""
    rows = []
    grid = [(100, 10, 30), (500, 20, 30)]
    for m, n, reps in grid:
        for rep in range(reps):
            rng = np.random.default_rng(7000 * m + rep)
            P, x0 = sphere_polar(m, n, rng)
            res = {
                "P": p_center(P, x0, EPS, MAX_SWEEPS),
                "CN": cn_center(P, x0, EPS, MAX_SWEEPS),
            }
            for name, r in res.items():
                rows.append([m, n, rep, name, r.sweeps, r.chords, r.seconds,
                             P.rel_centrality(r.x), int(r.converged)])
            for name, ordering in [("CN-rand", "random"),
                                   ("CN-greedy", "greedy")]:
                r = cn_center(P, x0, 0.0, BUDGET, ordering=ordering,
                              rng=np.random.default_rng(rep))
                xt = tail_average(r, 10)
                rows.append([m, n, rep, name, r.sweeps, r.chords, r.seconds,
                             P.rel_centrality(xt), 0])
            x_ana = P.analytic_center(x0)
            rows.append([m, n, rep, "analytic", 0, 0, 0,
                         P.rel_centrality(x_ana), 1])
        print(f"  e7: ({m},{n}) done")
    save_csv("e7_sphere.csv",
             ["m", "n", "rep", "method", "sweeps", "chords", "seconds",
              "relC_final", "converged"], rows)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    todo = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "e6"]
    t0 = time.perf_counter()
    for name in todo:
        print(f"=== {name} ===")
        globals()[name]()
    print(f"total {time.perf_counter() - t0:.1f}s")
