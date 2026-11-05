"""Compare pipeline output to the synthetic tile's known truth (only works on synthetic tiles)."""
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import shape

truth = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
props = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
to_utm = Transformer.from_crs(4326, 3160, always_xy=True)

houses = [f for f in props["features"] if not f["properties"].get("outbuilding")]
rows = []
for lot in truth["lots"]:
    tx, ty = lot["house_centroid_utm"]
    best, bd = None, 1e9
    for f in houses:
        c = shape(f["geometry"]).centroid
        x, y = to_utm.transform(c.x, c.y)
        d = np.hypot(x - tx, y - ty)
        if d < bd:
            best, bd = f["properties"], d
    rows.append((lot, best, bd))

print(f"{'lot':4} {'mgmt':>5} {'pid':4} {'off_m':>5} | {'z1 can true/est':>16} {'z1 lad true/est':>16} | "
      f"{'z2 can true/est':>16} {'z2 lad true/est':>16} | score rating")
err = {k: [] for k in ("zone1_canopy", "zone1_ladder", "zone2_canopy", "zone2_ladder")}
for lot, p, d in sorted(rows, key=lambda r: r[0]["management"]):
    cells = []
    for k in err:
        t, e = lot[f"{k}_pct_true"], p[f"{k}_pct"]
        err[k].append(e - t)
        cells.append(f"{t:6.1f} / {e:6.1f}")
    print(f"{lot['lot_id']:4} {lot['management']:5.2f} {p['property_id']:4} {d:5.1f} | "
          f"{cells[0]:>16} {cells[1]:>16} | {cells[2]:>16} {cells[3]:>16} | {p['overall_score']:5d} {p['rating']}")

print("\nmean abs error (percentage points):")
for k, v in err.items():
    print(f"  {k:14} MAE {np.mean(np.abs(v)):5.1f}  bias {np.mean(v):+5.1f}")
mg = [r[0]["management"] for r in rows]
sc = [r[1]["overall_score"] for r in rows]
print(f"\nSpearman(management, overall score) = {__import__('scipy').stats.spearmanr(mg, sc)[0]:+.2f}  (expect strongly negative)")
print(f"houses matched within 3 m: {sum(r[2] < 3 for r in rows)}/{len(rows)}")
