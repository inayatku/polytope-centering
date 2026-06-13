"""Generate LaTeX tables from ../results CSVs into ../tables, and print a
plain-text summary of the headline numbers for the paper text."""

import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "tables")
os.makedirs(TABLES, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        r = csv.reader(f)
        next(r)
        return list(r)


def write(name, text):
    path = os.path.join(TABLES, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote tables/{name}")


def ms(vals, fmt="{:.2f}"):
    vals = [v for v in vals if not np.isnan(v)]
    return (fmt.format(np.mean(vals)) + " $\\pm$ " +
            fmt.format(np.std(vals)))


# ------------------------------------------------------------- scaling table
def scaling():
    rows = load("e4_scaling.csv")
    D = defaultdict(list)
    for m, n, rep, meth, sweeps, chords, sec, rf, r5, r30, conv in rows:
        D[(int(m), int(n), meth)].append(
            (int(sweeps), int(chords), float(sec), float(rf),
             float(r5) if r5 != "nan" else np.nan,
             float(r30) if r30 != "nan" else np.nan, int(conv)))
    sizes = sorted({(k[0], k[1]) for k in D})
    methods = ["P", "CN", "HYB", "CN-rand", "CN-greedy", "analytic",
               "chebyshev"]
    labels = {"P": "Jacobi (P)", "CN": "recursive (CN)", "HYB": "hybrid",
              "CN-rand": "CN-rand (tail)", "CN-greedy": "CN-greedy (tail)",
              "analytic": "analytic center", "chebyshev": "Chebyshev LP"}

    lines = [
        "\\begin{table}[!htbp]\\centering",
        "\\caption{Scaling study on random simplex-like polytopes "
        "(30 instances per size; 10 for $m{=}1000$), corner start, "
        "$\\varepsilon=10^{-3}$. Jacobi (P) = Moretti's P-center sweep; "
        "recursive (CN) = the CN-center sweep; hybrid = CN$\\to$P switch; "
        "CN-rand / CN-greedy = randomized / greedy orderings on a fixed "
        "30-sweep budget with 10-sweep tail averaging (no displacement "
        "stopping, so no sweep count). `sweeps' is sweeps to "
        "$\\Delta<\\varepsilon$ ($m$ chords per sweep); rel.\\ $C$@5 is "
        "centrality after a 5-sweep budget; rel.\\ $C$ is centrality at the "
        "method's stopping point, relative to the Chebyshev radius. Time "
        "compares interpreted Python sweep loops against compiled solvers "
        "(HiGHS, vectorized Newton); chord counts are the primary work "
        "measure.}",
        "\\label{tab:scaling}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "$(m,n)$ & method & sweeps & rel.\\ $C$@5 & rel.\\ $C$ final & "
        "time (s) \\\\",
        "\\midrule",
    ]
    summary = []
    for (m, n) in sizes:
        first = True
        for meth in methods:
            v = D.get((m, n, meth))
            if not v:
                continue
            sw = [a[0] for a in v]
            rf = [a[3] for a in v]
            sec = [a[2] for a in v]
            r5 = [a[4] for a in v]
            tag = f"$({m},{n})$" if first else ""
            first = False
            sw_txt = ("--" if meth in ("CN-rand", "CN-greedy", "analytic",
                                       "chebyshev")
                      else ms(sw, "{:.1f}"))
            r5_txt = ("--" if meth in ("analytic", "chebyshev")
                      else ms(r5))
            rf_txt = "1.00 (exact)" if meth == "chebyshev" else ms(rf)
            lines.append(f"{tag} & {labels[meth]} & {sw_txt} & {r5_txt} & "
                         f"{rf_txt} & {ms(sec, '{:.3f}')} \\\\")
            summary.append((m, n, meth, np.mean(sw), np.mean(rf),
                            np.mean(sec)))
        lines.append("\\midrule" if (m, n) != sizes[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]
    write("table_scaling.tex", "\n".join(lines) + "\n")

    print("\n--- scaling summary (means) ---")
    for m, n, meth, sw, rf, sec in summary:
        print(f"({m:5d},{n:3d}) {meth:10s} sweeps={sw:8.1f} relC={rf:.3f} "
              f"t={sec:.4f}s")


# ---------------------------------------------------------- redundancy table
def redundancy():
    rows = load("e5_redundancy.csv")
    D = defaultdict(list)
    for rep, copies, meth, drift in rows:
        D[(int(copies), meth)].append(float(drift))
    methods = ["chebyshev", "analytic", "P", "P-w", "CN", "CN-w"]
    labels = {"chebyshev": "Chebyshev", "analytic": "analytic",
              "P": "P-center", "P-w": "P-center (mult.\\ weights)",
              "CN": "CN-center", "CN-w": "CN-center (mult.\\ weights)"}
    lines = [
        "\\begin{table}[!htbp]\\centering",
        "\\caption{Redundancy stress test: half of the constraints of 30 "
        "random $50\\times10$ polytopes are duplicated $\\times2$ / "
        "$\\times5$. Entries are the drift of the computed center, "
        "$\\|x_{dup}-x_{orig}\\|/r_{cheb}$ (mean $\\pm$ std).}",
        "\\label{tab:redundancy}",
        "\\small",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "method & duplication $\\times2$ & duplication $\\times5$ \\\\",
        "\\midrule",
    ]
    print("\n--- redundancy summary ---")
    for meth in methods:
        c2 = D[(2, meth)]
        c5 = D[(5, meth)]
        lines.append(f"{labels[meth]} & {ms(c2, '{:.4f}')} & "
                     f"{ms(c5, '{:.4f}')} \\\\")
        print(f"{meth:10s} x2: {np.mean(c2):.4f}   x5: {np.mean(c5):.4f}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_redundancy.tex", "\n".join(lines) + "\n")


# ------------------------------------------------------- sphere-family table
def sphere():
    rows = load("e7_sphere.csv")
    D = defaultdict(list)
    for m, n, rep, meth, sw, ch, sec, rf, conv in rows:
        D[(int(m), int(n), meth)].append((int(sw), float(rf)))
    methods = ["P", "CN", "CN-rand", "CN-greedy", "analytic"]
    lines = [
        "\\begin{table}[!htbp]\\centering",
        "\\caption{Replication on the second instance family (rows uniform "
        "on the unit sphere, $b=\\mathbf{1}$; round geometry, near-boundary "
        "start; 30 instances per size). Same protocol as "
        "Table~\\ref{tab:scaling}.}",
        "\\label{tab:sphere}",
        "\\small",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "$(m,n)$ & method & sweeps & rel.\\ $C$ final \\\\",
        "\\midrule",
    ]
    sizes = [(100, 10), (500, 20)]
    for (m, n) in sizes:
        first = True
        for meth in methods:
            v = D.get((m, n, meth))
            if not v:
                continue
            sw = [a[0] for a in v]
            rf = [a[1] for a in v]
            tag = f"$({m},{n})$" if first else ""
            first = False
            sw_txt = ("--" if meth in ("CN-rand", "CN-greedy", "analytic")
                      else ms(sw, "{:.1f}"))
            lines.append(f"{tag} & {meth} & {sw_txt} & {ms(rf)} \\\\")
        lines.append("\\midrule" if (m, n) != sizes[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]
    write("table_sphere.tex", "\n".join(lines) + "\n")


# ------------------------------------------------------------- order table
def order():
    rows = load("e6_order.csv")
    disp = [float(r[1]) for r in rows]
    diam = [float(r[2]) for r in rows]
    relm = [float(r[4]) for r in rows]
    relt = [float(r[5]) for r in rows]
    txt = (
        "% Order-dependence summary (E6), 10 instances x 30 permutations\n"
        f"% mean dispersion / r_cheb: {np.mean(disp):.4f} "
        f"(+/- {np.std(disp):.4f})\n"
        f"% mean diameter  / r_cheb: {np.mean(diam):.4f}\n"
        f"% mean relC of CN limits:  {np.mean(relm):.3f}\n"
        f"% relC of tail-avg randomized: {np.mean(relt):.3f}\n"
    )
    write("order_summary.tex", txt)
    print("\n--- order dependence ---")
    print(txt)


if __name__ == "__main__":
    scaling()
    redundancy()
    sphere()
    order()
