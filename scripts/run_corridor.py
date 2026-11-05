"""
Score every house across many 1 km LiDAR tiles (a whole road corridor) without one giant grid.

Each tile is processed on its own grid with a 50 m border borrowed from its neighbours
(read with fast COPC spatial queries), so houses near tile edges still get their full
30 m ring. Only houses whose centre falls inside the tile are kept, so nothing is
counted twice. Memory stays at about one tile per worker, however many tiles there are.

    python scripts/run_corridor.py --tiles "D:/lidar/*.laz" --out output_corridor ^
        --config config_ssm2024.yaml --name "Searchmont corridor" --workers 2

Then build the map:
    python scripts/build_map.py --out output_corridor --html output_corridor/map.html ...
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
import shapely
from pyproj import CRS
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firesmart_lidar import pipeline as P   # noqa: E402
from firesmart_lidar import export as E     # noqa: E402

HALO = 50.0
TILE_RE = re.compile(r"1km\d{2}(\d{3})0(\d{4})0")


# ----------------------------------------------------------------- tile index
def index_tiles(paths):
    """{(e_km, n_km): path}. Uses the Ontario tile name when present, else the header."""
    idx = {}
    for p in paths:
        m = TILE_RE.search(p.name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
        else:
            with laspy.open(p) as f:
                key = (int(f.header.mins[0] // 1000), int(f.header.mins[1] // 1000))
        idx.setdefault(key, p)
    return idx


def header_crs(path, cfg):
    with laspy.open(path) as f:
        try:
            crs = f.header.parse_crs()
        except Exception:
            crs = None
    if crs is not None and crs.is_compound:
        crs = crs.sub_crs_list[0]
    if crs is None:
        crs = CRS.from_epsg(cfg["crs"]["source_epsg"])
    if crs.to_epsg() is None and crs.to_epsg(min_confidence=50):
        crs = CRS.from_epsg(crs.to_epsg(min_confidence=50))
    return crs


def read_box(path, x0, y0, x1, y1, cfg):
    """Points of one file inside a box, as the dict the pipeline uses."""
    try:
        with laspy.CopcReader.open(path) as r:
            pts = r.query(laspy.copc.Bounds(mins=np.array([x0, y0]), maxs=np.array([x1, y1])))
    except Exception:                       # plain LAS/LAZ: read all, then crop
        pts = laspy.read(path).points
    x, y = np.asarray(pts.x), np.asarray(pts.y)
    cls = np.asarray(pts.classification)
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & ~np.isin(cls, cfg["classes"]["noise"])
    return {"x": x[keep], "y": y[keep], "z": np.asarray(pts.z)[keep], "cls": cls[keep],
            "rn": np.asarray(pts.return_number)[keep], "nr": np.asarray(pts.number_of_returns)[keep]}


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ----------------------------------------------------------------- one tile
def process_tile(key, idx, cfg, px):
    e, n = key
    tx0, ty0 = e * 1000.0, n * 1000.0
    x0, y0, x1, y1 = tx0 - HALO, ty0 - HALO, tx0 + 1000 + HALO, ty0 + 1000 + HALO
    parts = []
    for de in (-1, 0, 1):
        for dn in (-1, 0, 1):
            nb = idx.get((e + de, n + dn))
            if nb is None:
                continue
            nx0, ny0 = (e + de) * 1000.0, (n + dn) * 1000.0
            bx0, by0, bx1, by1 = max(x0, nx0), max(y0, ny0), min(x1, nx0 + 1000), min(y1, ny0 + 1000)
            if bx0 < bx1 and by0 < by1:
                parts.append(read_box(nb, bx0, by0, bx1, by1, cfg))
    pts = concat(parts)
    cell = cfg["raster"]["cell_m"]
    grid = P.Grid(x0, y0, cell, int((x1 - x0) / cell), int((y1 - y0) / cell))

    dtm, nonground = P.build_dtm(pts, grid, cfg["classes"]["ground"])
    hag = P.height_above_ground(pts, grid, dtm)
    roof, method = P.detect_roof_cells(pts, hag, nonground, grid, cfg)
    footprints = P.footprints_from_mask(roof, grid, cfg)
    GX, GY = grid.centers()
    bld_mask = np.zeros((grid.ny, grid.nx), bool)
    for f in footprints:
        q = f.buffer(0.5)
        mnx, mny, mxx, mxy = q.bounds
        r0, c0 = grid.rc(np.array([mnx]), np.array([mny]))
        r1, c1 = grid.rc(np.array([mxx]), np.array([mxy]))
        sub = (slice(r0[0], r1[0] + 1), slice(c0[0], c1[0] + 1))
        bld_mask[sub] |= shapely.contains_xy(q, GX[sub], GY[sub])
    chm, ladder_n, overhang = P.vegetation_rasters(pts, hag, nonground, bld_mask, grid, cfg)
    del pts, hag

    # coverage = tiles that actually exist around this one
    cov = shapely.union_all([box((e + de) * 1000, (n + dn) * 1000, (e + de + 1) * 1000, (n + dn + 1) * 1000)
                             for de in (-1, 0, 1) for dn in (-1, 0, 1) if (e + de, n + dn) in idx])
    core = box(tx0, ty0, tx0 + 1000, ty0 + 1000)
    # score only houses centred in this tile; outbuildings from the halo still count for distance
    in_core = [f for f in footprints if core.contains(f.centroid)]
    valid = shapely.contains_xy(cov, GX, GY)
    results, rings, _ = P.score_buildings(footprints, chm, ladder_n, overhang, dtm, grid, cfg,
                                          coverage=cov, valid=valid)
    results = [m for m in results if core.contains(m["_geom"].centroid)]
    keep_ids = {m["property_id"] for m in results}
    rings = [r for r in rings if r["property_id"] in keep_ids]
    outs = [f for f in in_core if f.area < cfg["buildings"]["primary_min_area_m2"]]
    for m in results:                       # local id -> tile-unique id, renumbered later
        m["_local"] = f"{e}_{n}_{m['property_id']}"
    for r in rings:
        r["_local"] = f"{e}_{n}_{r['property_id']}"

    # coarse overlay blocks for the core 1 km only
    h = int(round(HALO / cell))
    k = int(round(px / cell))
    def blocks(a, how):
        c = a[h:h + int(1000 / cell), h:h + int(1000 / cell)]
        c = c[: (c.shape[0] // k) * k, : (c.shape[1] // k) * k].reshape(c.shape[0] // k, k, c.shape[1] // k, k)
        return getattr(c, how)(axis=(1, 3))
    lad = (ladder_n >= cfg["vegetation"]["ladder_min_returns"]).astype(np.float32)
    ov = {"dtm": blocks(dtm, "mean").astype(np.float32), "chm": blocks(chm, "max").astype(np.float32),
          "lad": blocks(lad, "mean").astype(np.float32), "bld": blocks(bld_mask, "max")}
    return key, results, rings, outs, ov, method


def _work(args):
    key, idx, cfg, px = args
    t = time.time()
    out = process_tile(key, idx, cfg, px)
    return out + (time.time() - t,)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, nargs="+", help="LAS/LAZ/COPC files or glob patterns")
    ap.add_argument("--out", default="output_corridor", type=Path)
    ap.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml", type=Path)
    ap.add_argument("--name", default="")
    ap.add_argument("--px", default=3.0, type=float, help="map overlay pixel size in metres (3 keeps the map small)")
    ap.add_argument("--workers", default=1, type=int, help="tiles processed in parallel (~3 GB RAM each)")
    a = ap.parse_args()
    t0 = time.time()
    cfg = P.load_config(a.config)
    a.out.mkdir(parents=True, exist_ok=True)

    paths = sorted({Path(p) for pat in a.tiles for p in glob.glob(pat)})
    if not paths:
        sys.exit("no tiles matched")
    idx = index_tiles(paths)
    crs = header_crs(next(iter(idx.values())), cfg)
    keys = sorted(idx)
    print(f"{len(keys)} tiles  crs=EPSG:{crs.to_epsg()}  workers={a.workers}")

    E0, N0 = min(k[0] for k in keys), min(k[1] for k in keys)
    E1, N1 = max(k[0] for k in keys) + 1, max(k[1] for k in keys) + 1
    per = int(1000 / a.px)
    shape = ((N1 - N0) * per, (E1 - E0) * per)
    g_dtm = np.full(shape, np.nan, np.float32); g_chm = np.zeros(shape, np.float32)
    g_lad = np.zeros(shape, np.float32); g_bld = np.zeros(shape, bool); g_ok = np.zeros(shape, bool)

    all_res, all_rings, all_outs, methods = [], [], [], set()
    jobs = [(k, idx, cfg, a.px) for k in keys]
    done = 0
    if a.workers > 1:
        ex = ProcessPoolExecutor(a.workers)
        it = as_completed([ex.submit(_work, j) for j in jobs])
        get = lambda f: f.result()          # noqa: E731
    else:
        it = jobs
        get = _work
    for item in it:
        key, res, rings, outs, ov, method, dt = get(item)
        done += 1
        e, n = key
        c0 = (e - E0) * per
        rs = (n - N0) * per
        for name, arr in (("dtm", g_dtm), ("chm", g_chm), ("lad", g_lad), ("bld", g_bld)):
            arr[rs:rs + per, c0:c0 + per] = ov[name][:per, :per]
        g_ok[rs:rs + per, c0:c0 + per] = True
        all_res += res; all_rings += rings; all_outs += outs; methods.add(method)
        eta = (time.time() - t0) / done * (len(keys) - done)
        print(f"  [{done:3d}/{len(keys)}] tile {e}/{n}: {len(res):3d} houses  {dt:5.1f}s  (eta {eta/60:4.1f} min)", flush=True)

    # global ids, north to south then west to east
    all_res.sort(key=lambda m: (-round(m["_geom"].centroid.y, -1), m["_geom"].centroid.x))
    width = max(3, len(str(len(all_res))))
    remap = {}
    for i, m in enumerate(all_res, start=1):
        remap[m["_local"]] = f"P{i:0{width}d}"
        m["property_id"] = remap[m["_local"]]
    for r in all_rings:
        r["property_id"] = remap[r["_local"]]

    E.write_vectors(all_res, all_rings, all_outs, crs, a.out)
    g_dtm = np.where(np.isnan(g_dtm), np.nanmean(g_dtm), g_dtm)
    grid = P.Grid(E0 * 1000.0, N0 * 1000.0, a.px, shape[1], shape[0])
    lad_n = (g_lad >= 0.25).astype(np.int64) * cfg["vegetation"]["ladder_min_returns"]
    meta = E.write_rasters(g_dtm, g_chm, lad_n, g_bld, grid, crs, cfg, a.out,
                           px_m=a.px, canopy_fmt="webp", valid=g_ok)
    summary = {
        "source": [p.name for p in paths], "area_name": a.name, "crs": f"EPSG:{crs.to_epsg()}",
        "tiles": len(keys), "building_method": ", ".join(sorted(methods)),
        "houses": len(all_res), "outbuildings": len(all_outs),
        "ratings": {r: sum(m["rating"] == r for m in all_res) for r in cfg["scoring"]["ratings"]},
        "config": cfg, "raster": meta, "runtime_s": round(time.time() - t0, 1),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["ratings"]), f"{len(all_res)} houses  done in {summary['runtime_s'] / 60:.1f} min -> {a.out}/")


if __name__ == "__main__":
    main()
