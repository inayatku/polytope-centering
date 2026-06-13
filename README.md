# Follow-up paper: "Projection-Based Centering of Polytopes Revisited"

Code and data reproduction package for the paper *"Projection-Based
Centering of Polytopes Revisited: Work-Fair Comparisons, Geometry
Dependence, and Order-Robust Accelerated Variants."* It builds on *A New
Technique for Determining Approximate Center of a Polytope* (Inayatullah et
al., *Advances in Operations Research*, 2019) with fixed-point theory,
work-fair experiments, and new algorithmic variants.

This repository contains the Python implementation, all experiment scripts,
raw results, figures, and tables; every number, figure, and table in the
paper reproduces from fixed random seeds. The manuscript itself is not
included here.

## Layout

```
followup_paper/
  code/        Python implementation + experiments (Python 3.11, numpy, scipy, matplotlib)
    polytope.py        polytope class: chords (O(m) via Gram matrix), centrality,
                       Chebyshev (LP), analytic center (Newton), generators
    centers.py         p_center (Jacobi), cn_center (recursive; orderings, blocks,
                       weights, SOR), hybrid_center (CN->P), tail_average
    test_sanity.py     22 verification checks - run this first
    pilot.py           parameter calibration (omega grid, sweep costs)
    probe_geometry.py  thin-triangle / anisotropy probes
    probe_wedge.py     validation against the original paper's wedge polytope
    probe_hybrid.py    diagnosis of hybrid landing quality (stopping-rule artifact)
    probe_order_verify.py  verifies E6 order-dependence is not a censoring artifact
    experiments.py     E1-E6 -> ../results/*.csv
    make_figures.py    ../results -> ../figures/fig1..fig3 (.pdf/.png)
    make_tables.py     ../results -> ../tables/*.tex + console summary
    summarize_e3.py    console summary of the omega ablation
  results/     CSVs (raw experimental data)
  figures/     fig1_trajectories, fig2_convergence, fig3_omega
  tables/      table_scaling.tex, table_redundancy.tex, table_sphere.tex, order_summary.tex
```

## Reproduce everything

```powershell
cd code
python test_sanity.py     # 22 checks must pass
python experiments.py     # E1-E6, ~20-30 min total
python make_figures.py
python make_tables.py
```

## Experiments

| ID | Question | Output |
|----|----------|--------|
| E1 | What do the trajectories look like (wedge narrow/wide start, random 2-D)? | fig1 |
| E2 | Quality vs chord evaluations for all methods | fig2 |
| E3 | Where does (over/under-)relaxation pay? | fig3 |
| E4 | Scaling over (m,n) grid, all methods + LP/Newton baselines | table_scaling |
| E5 | Drift under duplicated constraints | table_redundancy |
| E6 | How order-dependent is the CN limit? | order_summary |

## Headline findings (as of 2026-06-12)

1. Work-fair accounting (chords, uniform sweep-level stopping): CN's advantage
   over P is real but geometry-specific — wedge/narrow-corner: CN 40 vs P 62
   sweeps; wide start: tie (23 vs 22); benign random instances: P is several
   times faster AND more central (e.g. 100x10: P 10.5 sweeps relC 0.79 vs
   CN 51.2 sweeps relC 0.62).
2. The mechanism: CN's first sub-step jumps to the first chord midpoint,
   discarding the incumbent — ideal for escaping narrow corners, destabilizing
   otherwise.
3. Centrality peaks BEFORE convergence on random instances: early stopping
   gives better centers than running to tolerance.
4. Over-relaxation (w=1.8) pays exactly in slow regimes (wedge -42%,
   aniso -35% sweeps); under-relaxation (w=0.6) damps CN's oscillation on
   benign instances (53.5 -> 34.7 sweeps).
5. Multiplicity weights make the weighted P-sweep EXACTLY invariant to
   duplicated constraints (Prop. 3; verified to 2e-16); reduces CN drift.
6. CN's limit is STRONGLY order-dependent: across 30 fixed orderings on
   random 50x10 instances the limits scatter with dispersion 2.39+-0.27
   r_cheb and diameter 6.57 r_cheb (several Chebyshev radii apart!), with
   mean relC only 0.34. Not a censoring artifact: 97.3% of the 300 runs
   fully converge under a 20,000-sweep cap (mean ~6,900 sweeps), and 1e-8
   continuation moves limits <= ~0.05 r_cheb. Randomized ordering + tail
   averaging gives one ordering-free point of BETTER centrality (0.42)
   from a 60-sweep budget.
7. Greedy min-slack ordering attains the best budget-stopped centrality
   (relC ~0.95 in tests) but cycles, so it needs budget + tail averaging.
   On the sphere-polar replication family (E7) it is spectacular: relC
   0.977/0.964 from a 30-sweep budget, ahead of every converged method —
   its degradation at scale is specific to deep-corner simplex-like starts.
   E7 also replicates the P>CN ranking on a second generator (59 vs 155
   sweeps at 100x10).
8. Anisotropic instances slow both methods ~10x (neither is affine-invariant;
   analytic-center Newton is — future work: preconditioning).
9. Hybrid CN->P inherits the wedge transit speed of CN and the central,
   stable landing of P.

## Validation against the original paper

`probe_wedge.py` reconstructs Polytope 1 from Figure 1 of the original paper:
our P needs 62 sweeps (paper: 56) and lands at x ~= (1.446, 3.27) (paper:
(1.4453, 3.4772)) — close given the vertex reconstruction is approximate.
CN beats P there (40 vs 62) and ties from the wide corner, matching the
paper's Figures 6-7 qualitatively. The 5x magnitude reported in the paper is
not reproduced under a uniform sweep-level stopping rule.
