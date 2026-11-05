"""
Generate a synthetic airborne LiDAR tile that mimics Ontario's classified point cloud.

Why synthetic: lets the pipeline be built and *validated* against known truth before
pointing it at real GeoHub tiles. The output deliberately matches the real data's quirks:

  * LAS 1.4, point format 6, NAD83(CSRS) / UTM zone 16N (EPSG:3160)
  * Only ground (2) and noise (7) are classified; vegetation and buildings are left as
    class 1 "unclassified" -- as in many provincial deliveries -- so the pipeline has to
    find buildings from geometry rather than trusting a class code.
  * ~8 pulses/m^2 with multiple returns through canopy.

Scene: a rural road with 14 lots in boreal mixedwood (spruce/fir + birch/maple) near
Searchmont, ON. Each lot gets a "management" level from 0 (bush right up to the house)
to 1 (FireSmart-cleared yard, limbed trees). Truth is written to a JSON sidecar.

Usage:  python scripts/make_synthetic_tile.py --out data/synthetic_searchmont.laz
"""
import argparse
import json
from pathlib import Path

import laspy
import shapely
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely import affinity

RNG = np.random.default_rng(42)
CELL = 0.5                      # generation grid (m)
W, H = 640.0, 420.0             # tile size (m)
PULSES_PER_M2 = 8.0

# Tile origin: just west of Searchmont, ON, in EPSG:3160
X0, Y0 = Transformer.from_crs(4326, 3160, always_xy=True).transform(-84.083, 46.772)
X0, Y0 = float(np.floor(X0)), float(np.floor(Y0))

MANAGEMENT = [0.0, 0.9, 0.4, 1.0, 0.2, 0.6, 0.1,
              0.8, 0.3, 0.5, 0.95, 0.15, 0.7, 0.05]


def terrain(x, y):
    """Gentle valley side + a knoll, metres ASL."""
    return (330 + 0.045 * y - 0.01 * x
            + 14 * np.exp(-(((x - 470) / 90) ** 2 + ((y - 330) / 70) ** 2))
            + 1.2 * np.sin(x / 37.0) * np.cos(y / 53.0))


def smooth_noise(shape, scale_cells):
    from scipy.ndimage import gaussian_filter
    n = gaussian_filter(RNG.standard_normal(shape), scale_cells)
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


def build_lots():
    lots = []
    road_y = H / 2
    frontage = 88.0
    i = 0
    for side in (+1, -1):
        for k in range(7):
            cx = 40 + frontage * k + frontage / 2 + RNG.uniform(-8, 8)
            setback = RNG.uniform(28, 45)
            cy = road_y + side * (6 + setback)
            w, d = RNG.uniform(10, 16), RNG.uniform(8, 12)
            ang = RNG.uniform(-25, 25)
            house = affinity.rotate(box(cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2), ang)
            lot = {
                "lot_id": f"L{i + 1:02d}",
                "management": MANAGEMENT[i],
                "house": house,
                "house_angle": ang,
                "house_w": w, "house_d": d, "house_c": (cx, cy),
                "eave": RNG.uniform(3.0, 5.5),
                "pitch": RNG.uniform(0.25, 0.55),
                "outbuildings": [],
            }
            # some lots have a shed / garage; some too close to the house
            if RNG.random() < 0.6:
                dist = RNG.choice([9.0, 18.0, 25.0])
                ox = cx + RNG.choice([-1, 1]) * (w / 2 + dist)
                s = RNG.uniform(6, 8)
                lot["outbuildings"].append(box(ox - s / 2, cy - s / 2, ox + s / 2, cy + s / 2))
            lots.append(lot)
            i += 1
    return lots, road_y


def main(out_path: Path):
    nx, ny = int(W / CELL), int(H / CELL)
    xs = (np.arange(nx) + 0.5) * CELL
    ys = (np.arange(ny) + 0.5) * CELL
    GX, GY = np.meshgrid(xs, ys)

    lots, road_y = build_lots()

    # --- management field: how cleared each location is (distance to house based)
    from shapely import contains_xy, distance
    mgmt_field = np.zeros((ny, nx))
    house_dist = np.full((ny, nx), 1e9)
    for lot in lots:
        cx, cy = lot["house_c"]
        win = (np.abs(GX - cx) < 90) & (np.abs(GY - cy) < 90)
        pts = np.column_stack([GX[win], GY[win]])
        d = distance(lot["house"], shapely.points(pts))
        cur = house_dist[win]
        upd = d < cur
        house_dist[win] = np.where(upd, d, cur)
        m = lot["management"]
        clear_r = 10 + 25 * m                   # radius of FireSmart work
        effect = m * np.clip(1 - (d - clear_r) / 15, 0, 1)
        mf = mgmt_field[win]
        mgmt_field[win] = np.where(upd, effect, mf)

    road = np.abs(GY - road_y) < 4
    # driveways
    for lot in lots:
        cx, cy = lot["house_c"]
        y0, y1 = sorted([road_y, cy])
        road |= (np.abs(GX - cx) < 2.5) & (GY > y0) & (GY < y1)

    # --- trees
    density = 1 / 14.0                                   # stems per m^2 in bush
    n_trees = RNG.poisson(density * W * H)
    tx, ty = RNG.uniform(0, W, n_trees), RNG.uniform(0, H, n_trees)
    ti = np.clip((ty / CELL).astype(int), 0, ny - 1)
    tj = np.clip((tx / CELL).astype(int), 0, nx - 1)
    m_at = mgmt_field[ti, tj]
    hd_at = house_dist[ti, tj]
    conifer = RNG.random(n_trees) < 0.62
    # removal: managed yards keep few trees, and preferentially keep deciduous
    p_keep = 1 - m_at * np.where(conifer, 0.97, 0.75)
    keep = (RNG.random(n_trees) < p_keep) & ~road[ti, tj] & (hd_at > 1.0)
    # no stems inside (or hard against) sheds and garages
    for lot in lots:
        for ob in lot["outbuildings"]:
            keep &= ~shapely.contains_xy(ob.buffer(2.0), tx, ty)
    tx, ty, conifer, m_at = tx[keep], ty[keep], conifer[keep], m_at[keep]

    height = np.where(conifer, RNG.uniform(9, 21, tx.size), RNG.uniform(12, 23, tx.size))
    cbh = np.where(conifer, RNG.uniform(0.2, 1.8, tx.size), RNG.uniform(4, 8, tx.size))
    cbh = cbh + m_at * RNG.uniform(2.5, 4.0, tx.size)   # limbing up in managed yards
    cbh = np.minimum(cbh, height * 0.6)
    radius = np.where(conifer, RNG.uniform(1.4, 2.6, tx.size), RNG.uniform(2.5, 4.0, tx.size))

    top = np.zeros((ny, nx))
    base = np.full((ny, nx), np.inf)
    for x, y, h, b, r, c in zip(tx, ty, height, cbh, radius, conifer):
        j0, j1 = max(int((x - r) / CELL), 0), min(int((x + r) / CELL) + 1, nx)
        i0, i1 = max(int((y - r) / CELL), 0), min(int((y + r) / CELL) + 1, ny)
        if j0 >= j1 or i0 >= i1:
            continue
        d = np.hypot(GX[i0:i1, j0:j1] - x, GY[i0:i1, j0:j1] - y)
        inside = d < r
        if c:   # cone
            ztop = h - d * (h - b) / r
            zb = np.full_like(d, b)
        else:   # ellipsoid
            a = (h - b) / 2
            ctr = b + a
            half = a * np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
            ztop, zb = ctr + half, ctr - half
        sub_t, sub_b = top[i0:i1, j0:j1], base[i0:i1, j0:j1]
        top[i0:i1, j0:j1] = np.where(inside, np.maximum(sub_t, ztop), sub_t)
        base[i0:i1, j0:j1] = np.where(inside, np.minimum(sub_b, zb), sub_b)
    base[np.isinf(base)] = 0

    # --- shrub / regen layer (a key ladder fuel), suppressed by management and roads
    shrub = 2.6 * smooth_noise((ny, nx), 6) * (1 - mgmt_field) ** 2
    shrub[shrub < 0.6] = 0
    shrub[road] = 0

    # --- buildings raster
    bld_roof = np.full((ny, nx), np.nan)
    from shapely import points as shp_points
    all_pts = shp_points(np.column_stack([GX.ravel(), GY.ravel()]))
    for lot in lots:
        cx, cy = lot["house_c"]
        win = ((np.abs(GX - cx) < 45) & (np.abs(GY - cy) < 45)).ravel()
        inside = contains_xy(lot["house"], GX.ravel()[win], GY.ravel()[win])
        idx = np.flatnonzero(win)[inside]
        # gable roof: height rises toward the ridge (long axis)
        ang = np.radians(lot["house_angle"])
        px, py = GX.ravel()[idx] - cx, GY.ravel()[idx] - cy
        across = np.abs(-px * np.sin(ang) + py * np.cos(ang))
        roof = lot["eave"] + lot["pitch"] * (lot["house_d"] / 2 - across)
        bld_roof.ravel()[idx] = roof
        shrub.ravel()[idx] = 0
        for ob in lot["outbuildings"]:
            ins = contains_xy(ob, GX.ravel()[win], GY.ravel()[win])
            bld_roof.ravel()[np.flatnonzero(win)[ins]] = 3.0
            shrub.ravel()[np.flatnonzero(win)[ins]] = 0
    is_bld = ~np.isnan(bld_roof)
    # trees can't grow through roofs, but can overhang them
    top_over = np.where(is_bld & (top <= np.nan_to_num(bld_roof) + 1), 0, top)
    top = top_over

    # --- simulate pulses
    n_per_cell = PULSES_PER_M2 * CELL * CELL
    counts = RNG.poisson(n_per_cell, size=(ny, nx))
    ci = np.repeat(np.arange(ny * nx), counts.ravel())
    px = GX.ravel()[ci] + RNG.uniform(-CELL / 2, CELL / 2, ci.size)
    py = GY.ravel()[ci] + RNG.uniform(-CELL / 2, CELL / 2, ci.size)
    g = terrain(px, py)
    t_c, b_c, s_c = top.ravel()[ci], base.ravel()[ci], shrub.ravel()[ci]
    roof_c = bld_roof.ravel()[ci]
    has_tree = t_c > 0.5
    has_shrub = s_c > 0
    bld = ~np.isnan(roof_c)

    rx, ry, rz, rcls, rnum, rtot = [], [], [], [], [], []
    pulse_returns = []

    # each pulse -> list of (z_hag, cls) in order
    u = RNG.random((ci.size, 4))
    r1 = np.where(has_tree, t_c - np.minimum(RNG.exponential(0.6, ci.size), t_c - b_c), np.nan)
    r2 = np.where(has_tree & (u[:, 0] < 0.65), RNG.uniform(b_c, np.maximum(t_c, b_c + 0.1)), np.nan)
    r3 = np.where(has_shrub & (u[:, 1] < np.where(has_tree, 0.35, 0.8)),
                  RNG.uniform(0.15, np.maximum(s_c, 0.2)), np.nan)
    p_ground = np.where(has_tree, 0.35, np.where(has_shrub, 0.65, 1.0))
    rg = np.where(~bld & (u[:, 2] < p_ground), 0.0 + RNG.normal(0, 0.03, ci.size), np.nan)
    rroof = np.where(bld, roof_c + RNG.normal(0, 0.03, ci.size), np.nan)

    stack = np.column_stack([r1, r2, r3, rroof, rg])           # already in descending order
    cls_stack = np.array([1, 1, 1, 1, 2])
    valid = ~np.isnan(stack)
    # ensure every pulse yields at least one return (ground if nothing else)
    none = ~valid.any(1)
    stack[none, 4] = 0.0
    valid[none, 4] = True
    nret = valid.sum(1)
    order = np.cumsum(valid, 1)
    rows, cols = np.nonzero(valid)
    rx = px[rows]
    ry = py[rows]
    rz = g[rows] + stack[rows, cols]
    rcls = cls_stack[cols]
    rnum = order[rows, cols]
    rtot = nret[rows]

    # a few noise points (birds / multipath), classified 7
    k = 60
    rx = np.concatenate([rx, RNG.uniform(0, W, k)])
    ry = np.concatenate([ry, RNG.uniform(0, H, k)])
    rz = np.concatenate([rz, terrain(rx[-k:], ry[-k:]) + RNG.choice([-25, 80], k)])
    rcls = np.concatenate([rcls, np.full(k, 7)])
    rnum = np.concatenate([rnum, np.ones(k, int)])
    rtot = np.concatenate([rtot, np.ones(k, int)])

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = [X0, Y0, 0]
    header.scales = [0.01, 0.01, 0.01]
    header.add_crs(CRS.from_epsg(3160))
    las = laspy.LasData(header)
    las.x, las.y, las.z = rx + X0, ry + Y0, rz
    las.classification = rcls.astype(np.uint8)
    las.return_number = rnum.astype(np.uint8)
    las.number_of_returns = rtot.astype(np.uint8)
    las.intensity = RNG.integers(50, 4000, rx.size).astype(np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_path)

    # --- truth: canopy cover (>=3 m) in FireSmart rings, from the generating rasters
    truth = []
    from shapely import contains_xy as cxy
    for lot in lots:
        h = lot["house"]
        rings = {
            "zone1": h.buffer(10).difference(h.buffer(1.5)),
            "zone2": h.buffer(30).difference(h.buffer(10)),
        }
        rec = {"lot_id": lot["lot_id"], "management": lot["management"],
               "house_centroid_utm": [h.centroid.x + X0, h.centroid.y + Y0],
               "outbuilding_min_dist_m": (round(min(h.distance(o) for o in lot["outbuildings"]), 1)
                                          if lot["outbuildings"] else None)}
        for zn, ring in rings.items():
            mnx, mny, mxx, mxy = ring.bounds
            win = (GX > mnx) & (GX < mxx) & (GY > mny) & (GY < mxy)
            ins = cxy(ring, GX[win], GY[win])
            tops = top[win][ins]
            bases = base[win][ins]
            sh = shrub[win][ins]
            rec[f"{zn}_canopy_pct_true"] = round(100 * float((tops >= 3).mean()), 1)
            ladder = ((tops >= 3) & (bases <= 3.5)) | (sh >= 1.0)
            rec[f"{zn}_ladder_pct_true"] = round(100 * float(ladder.mean()), 1)
        truth.append(rec)
    with open(out_path.with_suffix(".truth.json"), "w") as f:
        json.dump({"crs": "EPSG:3160", "origin": [X0, Y0], "lots": truth}, f, indent=2)

    print(f"wrote {out_path}  points={len(las.points):,}  trees={tx.size:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_searchmont.laz", type=Path)
    main(ap.parse_args().out)
