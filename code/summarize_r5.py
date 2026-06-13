import csv
from collections import defaultdict

import numpy as np

rows = list(csv.reader(open("../results/e4_scaling.csv")))[1:]
D = defaultdict(list)
for m, n, rep, meth, sw, ch, sec, rf, r5, r30, conv in rows:
    if r5 != "nan":
        D[(int(m), int(n), meth)].append(float(r5))
for key in sorted(D):
    if key[2] in ("P", "CN", "HYB"):
        print(key, f"relC@5 = {np.mean(D[key]):.3f} +- {np.std(D[key]):.3f}")
