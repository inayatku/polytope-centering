import csv
from collections import defaultdict

import numpy as np

rows = list(csv.reader(open("../results/e7_sphere.csv")))[1:]
D = defaultdict(list)
for m, n, rep, meth, sw, ch, sec, rf, conv in rows:
    D[(int(m), int(n), meth)].append((int(sw), float(rf), int(conv)))
for (m, n) in [(100, 10), (500, 20)]:
    for meth in ["P", "CN", "CN-rand", "CN-greedy", "analytic"]:
        v = D[(m, n, meth)]
        sw = [a[0] for a in v]
        rf = [a[1] for a in v]
        cv = [a[2] for a in v]
        print(f"({m},{n}) {meth:10s} sweeps={np.mean(sw):7.1f}+-{np.std(sw):5.1f} "
              f"relC={np.mean(rf):.3f}+-{np.std(rf):.3f} conv={np.mean(cv):.0%}")
