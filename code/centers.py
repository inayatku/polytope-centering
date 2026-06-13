"""Center-finding iterations: P-center (Moretti 2003), CN-center
(Inayatullah et al.), and the new accelerated variants studied in the
follow-up paper (over-relaxation, randomized/greedy ordering, block hybrid,
weighted midpoints).

All solvers return a Result with the same work accounting:
- `chords`  : number of chord evaluations (the O(m) primitive; m per sweep)
- `sweeps`  : number of outer iterations (kept for comparison with the
              original paper, but chords/time are the fair measures)
- `history` : list of (chords_so_far, x_copy) snapshots, one per sweep
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from polytope import Polytope


@dataclass
class Result:
    x: np.ndarray
    sweeps: int
    chords: int
    seconds: float
    converged: bool
    history: list = field(default_factory=list)


def _sor_step(P: Polytope, x_prev: np.ndarray, x_sweep: np.ndarray,
              omega: float) -> np.ndarray:
    """x_prev + omega*(x_sweep - x_prev). For omega <= 1 the step is a convex
    combination of interior points, hence interior. For omega > 1 backtrack
    omega toward 1 until the candidate is strictly interior."""
    if omega == 1.0:
        return x_sweep
    if omega < 1.0:
        return x_prev + omega * (x_sweep - x_prev)
    w = omega
    while w > 1.0:
        cand = x_prev + w * (x_sweep - x_prev)
        if P.is_interior(cand):
            return cand
        w = 1.0 + 0.5 * (w - 1.0)
        if w - 1.0 < 1e-3:
            break
    return x_sweep


def p_center(P: Polytope, x0: np.ndarray, eps: float = 1e-3,
             max_sweeps: int = 2000, omega: float = 1.0,
             weights: np.ndarray | None = None) -> Result:
    """Moretti's projection center: Jacobi-style sweep (all m chord midpoints
    computed from the same point, then averaged). omega=1 reproduces the
    original method; omega>1 is the over-relaxed (SOR-P) variant."""
    x = np.asarray(x0, dtype=float).copy()
    w = np.ones(P.m) if weights is None else np.asarray(weights, float)
    chords = 0
    history = [(0, x.copy())]
    t0 = time.perf_counter()
    converged = False
    for k in range(1, max_sweeps + 1):
        s = P.slack(x)
        acc = np.zeros(P.n)
        for i in range(P.m):
            _, _, mid = P.chord(x, i, s)
            acc += w[i] * mid
            chords += 1
        x_sweep = acc / w.sum()
        x_new = _sor_step(P, x, x_sweep, omega)
        delta = np.max(np.abs(x_new - x))
        x = x_new
        history.append((chords, x.copy()))
        if delta < eps:
            converged = True
            break
    return Result(x, k, chords, time.perf_counter() - t0, converged, history)


def cn_center(P: Polytope, x0: np.ndarray, eps: float = 1e-3,
              max_sweeps: int = 2000, omega: float = 1.0,
              ordering: str = "fixed", block: int = 1,
              weights: np.ndarray | None = None,
              rng: np.random.Generator | None = None,
              perm: np.ndarray | None = None) -> Result:
    """CN-center sweep (Inayatullah et al.) and its variants.

    The original method (ordering='fixed', block=1, omega=1, weights=None)
    follows the paper exactly: at sub-step i the chord is shot from the
    latest running point x_{i-1}, and
        x_i = ( sum_{j<i} x_j + midpoint_i ) / i .

    ordering : 'fixed' (1..m), 'random' (fresh uniform permutation each
               sweep), 'greedy' (next = unprocessed constraint with smallest
               normalized slack at the current point), or 'given' (use perm).
    block    : Jacobi within blocks of this size, sequential across blocks
               (block=1 -> pure CN, block=m -> pure P).
    weights  : per-constraint weights applied to the running average
               (e.g. multiplicity weights for redundancy robustness).
    omega    : sweep-level over-relaxation as in p_center.
    """
    x = np.asarray(x0, dtype=float).copy()
    w = np.ones(P.m) if weights is None else np.asarray(weights, float)
    if ordering in ("random", "greedy") and rng is None:
        rng = np.random.default_rng(0)
    chords = 0
    history = [(0, x.copy())]
    t0 = time.perf_counter()
    converged = False
    for k in range(1, max_sweeps + 1):
        x_prev = x.copy()
        if ordering == "fixed":
            order = np.arange(P.m)
        elif ordering == "random":
            order = rng.permutation(P.m)
        elif ordering == "given":
            order = np.asarray(perm)
        elif ordering == "greedy":
            order = None  # chosen on the fly below
        else:
            raise ValueError(ordering)

        run_sum = np.zeros(P.n)   # weighted sum of previous running points
        run_w = 0.0
        xi = x.copy()
        remaining = set(range(P.m)) if ordering == "greedy" else None
        pos = 0
        while pos < P.m:
            if ordering == "greedy":
                s = P.slack(xi)
                rem = np.fromiter(remaining, dtype=int)
                i_block = [rem[np.argmin(s[rem] / P.row_norms[rem])]]
                remaining.discard(i_block[0])
                pos += 1
            else:
                i_block = order[pos:pos + block]
                pos += len(i_block)
            # Jacobi within the block: all midpoints from the same point xi,
            # combined into one weighted block-midpoint.
            s = P.slack(xi)
            blk_sum = np.zeros(P.n)
            blk_w = 0.0
            for i in i_block:
                _, _, mid = P.chord(xi, int(i), s)
                blk_sum += w[int(i)] * mid
                blk_w += w[int(i)]
                chords += 1
            blk_mid = blk_sum / blk_w
            # Paper's recursion (weighted form), one update per block:
            # x_i = (sum_{j<i} w_j x_j + w_B blk_mid) / (sum_{j<i} w_j + w_B).
            # block=1 reproduces the original CN-center exactly; block=m makes
            # xi the plain weighted midpoint average, i.e. one P-center sweep.
            new_w = run_w + blk_w
            xi = (run_sum + blk_w * blk_mid) / new_w
            run_sum += blk_w * xi
            run_w = new_w
        x_new = _sor_step(P, x_prev, xi, omega)
        delta = np.max(np.abs(x_new - x_prev))
        x = x_new
        history.append((chords, x.copy()))
        if delta < eps:
            converged = True
            break
    return Result(x, k, chords, time.perf_counter() - t0, converged, history)


def hybrid_center(P: Polytope, x0: np.ndarray, eps: float = 1e-3,
                  max_sweeps: int = 2000, switch_factor: float = 10.0) -> Result:
    """CN -> P hybrid: CN sweeps for the coarse transit (fast escape from
    narrow corners), switching to P sweeps for the fine landing (more central
    fixed point, stable stopping) once the CN sweep displacement falls below
    switch_factor * eps."""
    t0 = time.perf_counter()
    r1 = cn_center(P, x0, eps=switch_factor * eps, max_sweeps=max_sweeps)
    r2 = p_center(P, r1.x, eps=eps, max_sweeps=max(1, max_sweeps - r1.sweeps))
    history = r1.history + [(c + r1.chords, x) for c, x in r2.history[1:]]
    return Result(r2.x, r1.sweeps + r2.sweeps, r1.chords + r2.chords,
                  time.perf_counter() - t0, r2.converged, history)


def tail_average(result: Result, last: int = 10) -> np.ndarray:
    """Polyak-style tail average of the last `last` sweep outputs — the
    natural estimator for randomized/greedy orderings, whose iterates jitter
    near the limit and cannot satisfy an iterate-change stopping rule."""
    xs = [x for _, x in result.history[-last:]]
    return np.mean(xs, axis=0)
