"""Adversarial verification of E6: re-run all 300 fixed-ordering runs
(10 instances x 30 permutations, eps=1e-6, cap 20000) with a numpy
vectorization across the 30 runs of each instance, faithful to
cn_center(ordering='given', block=1, omega=1, weights=None).

Reports per-instance: the 5 fixed-ordering summary stats (to compare with
the archived e6_order.csv) plus converged fraction and sweep statistics.
"""

import numpy as np

from centers import cn_center, tail_average
from polytope import random_simplex_like

CAP = 20000
EPS = 1e-6


def run_instance(inst):
    rng = np.random.default_rng(600 + inst)
    P, x0 = random_simplex_like(50, 10, rng)
    _, r_cheb = P.chebyshev()
    m, n = P.m, P.n
    A, b, G = P.A, P.b, P.G
    R = 30
    perms = np.array([np.random.default_rng(p).permutation(m)
                      for p in range(R)])              # (R, m)
    XI = np.tile(np.asarray(x0, float), (R, 1))        # (R, n) current x
    X_rec = XI.copy()                                  # recorded limits
    sweeps_rec = np.full(R, CAP, dtype=int)
    conv = np.zeros(R, dtype=bool)

    posmask = G > 1e-14                                # (m, m): posmask[j, i]
    negmask = G < -1e-14

    for k in range(1, CAP + 1):
        X_prev = XI.copy()
        run_sum = np.zeros((R, n))
        for p in range(m):
            idx = perms[:, p]                          # (R,) constraint per run
            S = b[None, :] - XI @ A.T                  # (R, m) slacks
            D = G[:, idx].T                            # (R, m): row r = G[:, idx[r]]
            Pm = posmask[:, idx].T
            Nm = negmask[:, idx].T
            with np.errstate(divide="ignore", invalid="ignore"):
                t_plus = np.where(Pm, S / D, np.inf).min(axis=1)
                t_minus = np.where(Nm, S / (-D), np.inf).min(axis=1)
            mid = XI + (0.5 * (t_plus - t_minus))[:, None] * A[idx]
            XI = (run_sum + mid) / (p + 1.0)
            run_sum = run_sum + XI
        delta = np.max(np.abs(XI - X_prev), axis=1)
        newly = (~conv) & (delta < EPS)
        if np.any(newly):
            X_rec[newly] = XI[newly]
            sweeps_rec[newly] = k
            conv |= newly
        if conv.all():
            break
    X_rec[~conv] = XI[~conv]

    L = X_rec
    centroid_L = L.mean(axis=0)
    disp = np.linalg.norm(L - centroid_L, axis=1)
    diam = max(np.linalg.norm(a - c) for a in L for c in L)
    rr = cn_center(P, x0, eps=0.0, max_sweeps=60, ordering="random",
                   rng=np.random.default_rng(inst))
    xt = tail_average(rr, 20)
    return dict(inst=inst,
                mean_disp=disp.mean() / r_cheb,
                diam=diam / r_cheb,
                relC_of_mean=P.rel_centrality(centroid_L),
                mean_relC=float(np.mean([P.rel_centrality(x) for x in L])),
                relC_rand_tail=P.rel_centrality(xt),
                conv_frac=conv.mean(),
                n_unconv=int((~conv).sum()),
                min_sw=int(sweeps_rec.min()),
                mean_sw=float(sweeps_rec.mean()),
                max_sw=int(sweeps_rec.max()))


if __name__ == "__main__":
    import csv
    archived = {int(r["instance"]): r
                for r in csv.DictReader(open("../results/e6_order.csv"))}
    out = []
    for inst in range(10):
        d = run_instance(inst)
        a = archived[d["inst"]]
        print(f"inst {d['inst']}: disp {d['mean_disp']:.4f} "
              f"(csv {float(a['mean_disp_over_rcheb']):.4f})  "
              f"diam {d['diam']:.3f} (csv {float(a['diam_over_rcheb']):.3f})  "
              f"meanrelC {d['mean_relC']:.4f} "
              f"(csv {float(a['mean_relC']):.4f})  "
              f"conv {d['conv_frac']:.0%} unconv {d['n_unconv']}  "
              f"sweeps [{d['min_sw']}, {d['mean_sw']:.0f}, {d['max_sw']}]",
              flush=True)
        out.append(d)
    md = np.array([d["mean_disp"] for d in out])
    print(f"\nALL: mean disp {md.mean():.4f} +- {md.std():.4f}  "
          f"diam {np.mean([d['diam'] for d in out]):.4f}  "
          f"mean relC {np.mean([d['mean_relC'] for d in out]):.4f}  "
          f"rand tail {np.mean([d['relC_rand_tail'] for d in out]):.4f}")
    print(f"total unconverged runs: {sum(d['n_unconv'] for d in out)}/300  "
          f"global max sweeps among converged: "
          f"{max(d['max_sw'] for d in out)}")
