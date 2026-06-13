import csv
from collections import defaultdict

import numpy as np

rows = list(csv.reader(open("../results/e3_omega.csv")))[1:]
agg = defaultdict(list)
for g, m, o, s, r, c in rows:
    agg[(g, m, float(o))].append((int(s), float(r)))
for g in ["wedge", "rand100x10", "aniso50x5"]:
    for m in ["P", "CN"]:
        parts = []
        for o in [0.6, 1.0, 1.4, 1.8]:
            v = agg[(g, m, o)]
            parts.append(f"w={o}: {np.mean([a[0] for a in v]):7.1f} sw "
                         f"relC={np.mean([a[1] for a in v]):.3f}")
        print(f"{g:11s} {m:3s}: " + " | ".join(parts))
