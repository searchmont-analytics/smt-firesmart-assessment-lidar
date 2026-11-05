"""
Score every house across several adjacent LiDAR tiles as one seamless area.

Tiles are streamed one at a time (two passes), so memory stays at roughly one tile
regardless of how many you give it:
  pass 1  ground returns from every tile -> one DTM for the whole block
  pass 2  per tile: height above ground, building cells, canopy / ladder / roof-overhang
          rasters, all written into the shared grid
Houses near an inner tile edge now see their full 30 m ring from the neighbouring tile.

    python scripts/run_mosaic.py --tiles "path/to/tiles/*.laz" --out output_mosaic \
        --config config_ssm2024.yaml --name "Searchmont"
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import shapely
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firesmart_lidar import pipeline as P   # noqa: E402
from firesmart_lidar import export as E     # noqa: E402


def tile_bounds(paths):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for p in paths:
        with laspy.open(p) as f:
            h = f.header
            xs0.append(h.mins[0]); ys0.append(h.mins[1]); xs1.append(h.maxs[0]); ys1.append(h.maxs[1])
    return min(xs0), min(ys0), max(xs1), max(ys1)


def rasterize(polys, grid, buffer=0.0):
    GX, GY = grid.centers()
    mask = np.zeros((grid.ny, grid.nx), bool)
    for p in polys:
        q = p.buffer(buffer) if buffer else p
        mnx, mny, mxx, mxy = q.bounds
        r0, c0 = grid.rc(np.array([mnx]), np.array([mny]))
        r1, c1 = grid.rc(np.array([mxx]), np.array([mxy]))
        sub = (slice(r0[0], r1[0] + 1), slice(c0[0], c1[0] + 1))
        mask[sub] |= shapely.contains_xy(q, GX[sub], GY[sub])
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, nargs="+", help="LAS/LAZ files or glob patterns")
    ap.add_argument("--out", default="output_mosaic", type=Path)
    ap.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml", type=Path)
    ap.add_argument("--name", default="")
    ap.add_argument("--px", default=1.0, type=float, help="map overlay pixel size in metres")
    a = ap.parse_args()
    t0 = time.time()
    cfg = P.load_config(a.config)
    a.out.mkdir(parents=True, exist_ok=True)

    paths = sorted({Path(p) for pat in a.tiles for p in glob.glob(pat)})
    if not paths:
        sys.exit("no tiles matched")
    x0, y0, x1, y1 = tile_bounds(paths)
    cell = cfg["raster"]["cell_m"]
    grid = P.Grid(float(np.floor(x0)), float(np.floor(y0)), cell,
                  int(np.ceil((x1 - np.floor(x0)) / cell)), int(np.ceil((y1 - np.floor(y0)) / cell)))
    n = grid.nx * grid.ny
    print(f"{len(paths)} tiles  grid {grid.nx}x{grid.ny} m")

    # ---- pass 1: shared DTM
    gs = np.zeros(n); gk = np.zeros(n)
    crs, total = None, 0
    for p in paths:
        pts, c = P.read_points(p, cfg)
        crs = crs or c
        total += len(pts["x"])
        g = np.isin(pts["cls"], cfg["classes"]["ground"])
        r, cc = grid.rc(pts["x"][g], pts["y"][g])
        idx = r * grid.nx + cc
        gs += np.bincount(idx, weights=pts["z"][g], minlength=n)
        gk += np.bincount(idx, minlength=n)
        print(f"  pass 1  {p.name}  {len(pts['x']):,} pts")
        del pts
    dtm = np.where(gk > 0, gs / np.maximum(gk, 1), np.nan).reshape(grid.ny, grid.nx)
    holes = np.isnan(dtm)
    if holes.any():
        _, (ri, ci) = ndi.distance_transform_edt(holes, return_indices=True)
        dtm = dtm[ri, ci]
    dtm = ndi.gaussian_filter(dtm, 0.7)
    del gs, gk

    # ---- pass 2a: building cells from every tile
    roof = np.zeros((grid.ny, grid.nx), bool)
    method = None
    for p in paths:
        pts, _ = P.read_points(p, cfg)
        ng = ~np.isin(pts["cls"], cfg["classes"]["ground"])
        hag = P.height_above_ground(pts, grid, dtm)
        r_, method = P.detect_roof_cells(pts, hag, ng, grid, cfg)
        roof |= r_
        del pts, hag
    footprints = P.footprints_from_mask(roof, grid, cfg)
    bld_mask = rasterize(footprints, grid, buffer=0.5)
    print(f"buildings: {len(footprints)} (via {method})")

    # ---- pass 2b: vegetation rasters
    chm = np.zeros((grid.ny, grid.nx)); ladder_n = np.zeros((grid.ny, grid.nx), np.int64)
    overhang = np.zeros((grid.ny, grid.nx), bool)
    for p in paths:
        pts, _ = P.read_points(p, cfg)
        ng = ~np.isin(pts["cls"], cfg["classes"]["ground"])
        hag = P.height_above_ground(pts, grid, dtm)
        c_, l_, o_ = P.vegetation_rasters(pts, hag, ng, bld_mask, grid, cfg)
        np.maximum(chm, c_, out=chm); ladder_n += l_; overhang |= o_
        print(f"  pass 2  {p.name}")
        del pts, hag, c_, l_, o_

    results, rings, outs = P.score_buildings(footprints, chm, ladder_n, overhang, dtm, grid, cfg)
    print(f"scored {len(results)} houses, {len(outs)} outbuildings")

    E.write_vectors(results, rings, outs, crs, a.out)
    meta = E.write_rasters(dtm, chm, ladder_n, bld_mask, grid, crs, cfg, a.out, px_m=a.px, canopy_fmt="webp")
    summary = {
        "source": [p.name for p in paths], "area_name": a.name, "crs": f"EPSG:{crs.to_epsg()}",
        "points": int(total), "building_method": method, "houses": len(results), "outbuildings": len(outs),
        "ratings": {r: sum(m["rating"] == r for m in results) for r in cfg["scoring"]["ratings"]},
        "config": cfg, "raster": meta, "runtime_s": round(time.time() - t0, 1),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["ratings"]), f"done in {summary['runtime_s']} s -> {a.out}/")


if __name__ == "__main__":
    main()
