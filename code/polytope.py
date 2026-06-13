"""Polytope utilities for the CN-center follow-up paper.

A polytope is S = {x in R^n : A x <= b}, assumed bounded and full-dimensional
with a known strictly interior point.

Conventions used throughout:
- A "chord evaluation" at point x along constraint normal a_i means computing
  the two boundary intersections of the line {x + t a_i} with S. With the
  precomputed Gram matrix G = A A^T and the slack vector s = b - A x, one chord
  costs O(m). This is the unit of work reported in all experiments
  (both P-center and CN-center spend exactly m chords per sweep).
- centrality(x) = min_i (b_i - a_i^T x)/||a_i||  is the radius of the largest
  ball centered at x inscribed in S. Its maximum over x is the Chebyshev
  radius, so rel_centrality = centrality(x)/r_cheb in [0, 1] is a normalized,
  instance-independent quality measure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


class Polytope:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float).ravel()
        self.m, self.n = self.A.shape
        self.row_norms = np.linalg.norm(self.A, axis=1)
        if np.any(self.row_norms == 0):
            raise ValueError("zero constraint row")
        self.G = self.A @ self.A.T  # Gram matrix, column i = A @ a_i
        self._cheb = None

    # ----- basic queries -------------------------------------------------
    def slack(self, x: np.ndarray) -> np.ndarray:
        return self.b - self.A @ x

    def is_interior(self, x: np.ndarray, tol: float = 0.0) -> bool:
        return bool(np.all(self.slack(x) > tol))

    def centrality(self, x: np.ndarray) -> float:
        """Radius of the largest inscribed ball centered at x (<0 if x outside)."""
        return float(np.min(self.slack(x) / self.row_norms))

    def rel_centrality(self, x: np.ndarray) -> float:
        return self.centrality(x) / self.chebyshev()[1]

    # ----- chords ---------------------------------------------------------
    def chord(self, x: np.ndarray, i: int, s: np.ndarray | None = None):
        """Intersect the line {x + t * a_i} with S.

        Returns (t_plus, t_minus, midpoint) where t_plus = max{t : x + t a_i in S}
        and t_minus = max{t : x - t a_i in S}. Cost O(m) given the slack s.
        """
        if s is None:
            s = self.slack(x)
        d = self.G[:, i]  # = A @ a_i
        pos = d > 1e-14
        neg = d < -1e-14
        if not np.any(pos) or not np.any(neg):
            raise RuntimeError(
                "unbounded chord: polytope not bounded in direction of a_%d" % i
            )
        t_plus = np.min(s[pos] / d[pos])
        t_minus = np.min(s[neg] / (-d[neg]))
        delta = 0.5 * (t_plus - t_minus)
        return t_plus, t_minus, x + delta * self.A[i]

    def chord_length(self, t_plus: float, t_minus: float, i: int) -> float:
        return (t_plus + t_minus) * self.row_norms[i]

    # ----- reference centers ----------------------------------------------
    def chebyshev(self):
        """(center, radius) of the largest inscribed ball, via a single LP."""
        if self._cheb is None:
            c = np.zeros(self.n + 1)
            c[-1] = -1.0  # maximize r
            A_ub = np.hstack([self.A, self.row_norms[:, None]])
            res = linprog(c, A_ub=A_ub, b_ub=self.b,
                          bounds=[(None, None)] * self.n + [(0, None)],
                          method="highs")
            if not res.success:
                raise RuntimeError("Chebyshev LP failed: " + res.message)
            self._cheb = (res.x[:-1].copy(), float(res.x[-1]))
        return self._cheb

    def analytic_center(self, x0: np.ndarray, tol: float = 1e-10,
                        max_iter: int = 200):
        """Damped Newton on the log-barrier:  minimize -sum_i log(b_i - a_i^T x)."""
        x = np.asarray(x0, dtype=float).copy()
        if not self.is_interior(x):
            raise ValueError("analytic_center needs an interior start")
        for _ in range(max_iter):
            s = self.slack(x)
            g = self.A.T @ (1.0 / s)
            H = self.A.T @ (self.A / (s ** 2)[:, None])
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, -g, rcond=None)[0]
            lam2 = float(-g @ dx)  # Newton decrement squared
            if lam2 / 2.0 <= tol:
                break
            t = 1.0
            # backtracking: stay strictly interior, then Armijo on the barrier
            f0 = -np.sum(np.log(s))
            while not self.is_interior(x + t * dx):
                t *= 0.5
            while -np.sum(np.log(self.slack(x + t * dx))) > f0 - 0.25 * t * lam2:
                t *= 0.5
                if t < 1e-14:
                    break
            x = x + t * dx
        return x

    def centroid_2d(self):
        """Exact area centroid (n == 2 only), via the ordered vertex polygon."""
        V = self.vertices_2d()
        x, y = V[:, 0], V[:, 1]
        xs, ys = np.roll(x, -1), np.roll(y, -1)
        cross = x * ys - xs * y
        area = 0.5 * np.sum(cross)
        cx = np.sum((x + xs) * cross) / (6.0 * area)
        cy = np.sum((y + ys) * cross) / (6.0 * area)
        return np.array([cx, cy])

    def vertices_2d(self) -> np.ndarray:
        """Vertices of a 2-D polytope ordered counterclockwise."""
        assert self.n == 2
        from scipy.spatial import HalfspaceIntersection
        c, _ = self.chebyshev()
        hs = np.hstack([self.A, -self.b[:, None]])  # a^T x - b <= 0 format
        hi = HalfspaceIntersection(hs, c)
        V = hi.intersections
        ang = np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0])
        return V[np.argsort(ang)]


# ----- instance generators ---------------------------------------------------

def random_simplex_like(m: int, n: int, rng: np.random.Generator,
                        corner_start: bool = True):
    """Random bounded polytope {x >= 0, A0 x <= b0} with A0 > 0 (as in the
    original paper's MATLAB study; positivity of A0 guarantees boundedness).

    Returns (Polytope, x0) where x0 is a strictly interior starting point,
    placed near the origin corner when corner_start=True (the regime where
    the original paper reports the largest P/CN gap).
    """
    A0 = rng.uniform(0.1, 1.0, size=(m, n))
    b0 = rng.uniform(1.0, 2.0, size=m)
    A = np.vstack([A0, -np.eye(n)])
    b = np.concatenate([b0, np.zeros(n)])
    P = Polytope(A, b)
    t_max = np.min(b0 / A0.sum(axis=1))
    x0 = np.full(n, (0.02 if corner_start else 0.5) * t_max)
    assert P.is_interior(x0)
    return P, x0


def duplicate_constraints(P: Polytope, copies: int, rng: np.random.Generator,
                          frac: float = 1.0):
    """Return a new polytope where a fraction `frac` of constraints is
    duplicated `copies`-1 extra times (same geometry, redundant rows)."""
    k = max(1, int(round(frac * P.m)))
    idx = rng.choice(P.m, size=k, replace=False)
    A = [P.A]
    b = [P.b]
    for _ in range(copies - 1):
        A.append(P.A[idx])
        b.append(P.b[idx])
    return Polytope(np.vstack(A), np.concatenate(b))


def multiplicity_weights(P: Polytope, cos_tol: float = 0.999) -> np.ndarray:
    """Weight w_i = 1/(# constraints whose unit normal is nearly parallel to
    a_i's). Exact duplicates get down-weighted to sum to one chord's worth."""
    U = P.A / P.row_norms[:, None]
    C = U @ U.T
    counts = (C > cos_tol).sum(axis=1)
    return 1.0 / counts
