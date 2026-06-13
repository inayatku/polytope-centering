"""Sanity tests for polytope.py and centers.py.

Run:  python test_sanity.py
Every check prints PASS/FAIL; exits nonzero on any failure.
"""

import sys

import numpy as np

from polytope import (Polytope, duplicate_constraints, multiplicity_weights,
                      random_simplex_like)
from centers import cn_center, p_center

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def unit_square():
    # 0 <= x <= 1, 0 <= y <= 1
    A = np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([1.0, 0, 1, 0])
    return Polytope(A, b)


def main():
    sq = unit_square()

    # --- chord correctness on the unit square -----------------------------
    tp, tm, mid = sq.chord(np.array([0.25, 0.5]), 0)  # normal (1,0)
    check("chord t_plus on unit square", abs(tp - 0.75) < 1e-12, f"t+={tp}")
    check("chord t_minus on unit square", abs(tm - 0.25) < 1e-12, f"t-={tm}")
    check("chord midpoint on unit square",
          np.allclose(mid, [0.5, 0.5]), f"mid={mid}")

    # --- reference centers on the unit square -----------------------------
    c_cheb, r_cheb = sq.chebyshev()
    check("Chebyshev center of unit square",
          np.allclose(c_cheb, [0.5, 0.5], atol=1e-7) and abs(r_cheb - 0.5) < 1e-7,
          f"c={c_cheb}, r={r_cheb}")
    c_ana = sq.analytic_center(np.array([0.2, 0.7]))
    check("analytic center of unit square",
          np.allclose(c_ana, [0.5, 0.5], atol=1e-6), f"c={c_ana}")
    c_cent = sq.centroid_2d()
    check("centroid of unit square",
          np.allclose(c_cent, [0.5, 0.5], atol=1e-12), f"c={c_cent}")
    check("centrality at center = 0.5",
          abs(sq.centrality(np.array([0.5, 0.5])) - 0.5) < 1e-12)

    # --- symmetric body: P and CN converge to the center of symmetry ------
    x0 = np.array([0.05, 0.9])
    rp = p_center(sq, x0, eps=1e-10)
    rc = cn_center(sq, x0, eps=1e-10)
    check("P-center finds center of symmetric body",
          np.allclose(rp.x, [0.5, 0.5], atol=1e-6), f"x={rp.x}")
    check("CN-center finds center of symmetric body",
          np.allclose(rc.x, [0.5, 0.5], atol=1e-6), f"x={rc.x}")

    # --- balanced-chord (fixed point) residual at the P limit -------------
    rng = np.random.default_rng(42)
    P, x0 = random_simplex_like(25, 2, rng)
    rp = p_center(P, x0, eps=1e-12, max_sweeps=20000)
    s = P.slack(rp.x)
    resid = np.zeros(P.n)
    for i in range(P.m):
        tp, tm, _ = P.chord(rp.x, i, s)
        resid += (tp - tm) * P.A[i]
    check("balanced-chord condition at P fixed point",
          np.linalg.norm(resid) < 1e-6, f"||r||={np.linalg.norm(resid):.2e}")

    # --- all methods stay strictly interior, CN ~ P in quality ------------
    rc = cn_center(P, x0, eps=1e-10, max_sweeps=20000)
    check("P iterate strictly interior", P.is_interior(rp.x))
    check("CN iterate strictly interior", P.is_interior(rc.x))
    qp, qc = P.rel_centrality(rp.x), P.rel_centrality(rc.x)
    check("CN quality within 0.15 of P quality", abs(qp - qc) < 0.15,
          f"P={qp:.3f}, CN={qc:.3f}")

    # --- variant consistency ----------------------------------------------
    r1 = cn_center(P, x0, eps=1e-8)
    r2 = cn_center(P, x0, eps=1e-8, omega=1.0, block=1, ordering="fixed")
    check("default args reproduce plain CN", np.allclose(r1.x, r2.x))
    rb = cn_center(P, x0, eps=1e-8, block=P.m, max_sweeps=20000)
    rpp = p_center(P, x0, eps=1e-8, max_sweeps=20000)
    check("block=m CN equals P-center trajectory",
          np.allclose(rb.x, rpp.x, atol=1e-6),
          f"diff={np.max(np.abs(rb.x - rpp.x)):.2e}")
    rs = cn_center(P, x0, eps=1e-8, omega=1.5)
    check("SOR iterate strictly interior", P.is_interior(rs.x))

    # Randomized/greedy orderings change the sweep map every sweep, so the
    # iterate jitters near the limit and an iterate-change stopping rule
    # cannot fire. Evaluate them under a fixed budget + tail averaging.
    qc_ref = P.rel_centrality(cn_center(P, x0, eps=1e-10, max_sweeps=20000).x)
    rr = cn_center(P, x0, eps=0.0, max_sweeps=60, ordering="random",
                   rng=np.random.default_rng(1))
    tail = np.mean([x for _, x in rr.history[-20:]], axis=0)
    check("randomized CN interior + tail-avg quality ~ CN",
          P.is_interior(tail) and abs(P.rel_centrality(tail) - qc_ref) < 0.2,
          f"q={P.rel_centrality(tail):.3f} vs {qc_ref:.3f}")
    rg = cn_center(P, x0, eps=0.0, max_sweeps=60, ordering="greedy",
                   rng=np.random.default_rng(1))
    tail = np.mean([x for _, x in rg.history[-20:]], axis=0)
    check("greedy CN interior + tail-avg quality ~ CN",
          P.is_interior(tail) and abs(P.rel_centrality(tail) - qc_ref) < 0.2,
          f"q={P.rel_centrality(tail):.3f} vs {qc_ref:.3f}")

    # --- multiplicity weights vs exact duplication -------------------------
    # For the P-center the weighted average makes duplication exactly
    # invisible (identical trajectory). For CN the cancellation is only
    # approximate because the recursion is position-dependent.
    # Weights must be computed consistently on each polytope (random 2-D
    # normals can be near-parallel by chance, so the original polytope gets
    # nontrivial weights too). Exact triplication then makes the weighted
    # P trajectories on P and Pd identical.
    Pd = duplicate_constraints(P, copies=3, rng=np.random.default_rng(7))
    wP, wPd = multiplicity_weights(P), multiplicity_weights(Pd)
    rpw = p_center(Pd, x0, eps=1e-10, weights=wPd, max_sweeps=20000)
    rpu = p_center(P, x0, eps=1e-10, weights=wP, max_sweeps=20000)
    check("multiplicity weights exactly neutralize duplication (P)",
          np.linalg.norm(rpw.x - rpu.x) < 1e-9,
          f"drift={np.linalg.norm(rpw.x - rpu.x):.2e}")
    rw = cn_center(Pd, x0, eps=1e-10, weights=wPd, max_sweeps=20000)
    ru = cn_center(P, x0, eps=1e-10, weights=wP, max_sweeps=20000)
    rd = cn_center(Pd, x0, eps=1e-10, max_sweeps=20000)
    ru0 = cn_center(P, x0, eps=1e-10, max_sweeps=20000)
    drift_w = np.linalg.norm(rw.x - ru.x)
    drift_u = np.linalg.norm(rd.x - ru0.x)
    check("multiplicity weights reduce CN duplication drift",
          drift_w < drift_u,
          f"weighted={drift_w:.2e} vs unweighted={drift_u:.2e}")

    # --- work accounting ----------------------------------------------------
    check("chords == m * sweeps (P)", rp.chords == P.m * rp.sweeps)
    check("chords == m * sweeps (CN)", rc.chords == P.m * rc.sweeps)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
