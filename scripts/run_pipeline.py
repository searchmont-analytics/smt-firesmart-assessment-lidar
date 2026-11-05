"""
Score every house in a LiDAR tile against the FireSmart Home Ignition Zone.

    python scripts/run_pipeline.py --las data/synthetic_searchmont.laz --out output
    python scripts/run_pipeline.py --las path/to/geohub_tile.laz --footprints buildings.geojson

--footprints is optional: a GeoJSON of building polygons (any CRS GeoJSON allows, i.e. WGS84),
e.g. Ontario Building Footprints or Microsoft Canadian Building Footprints. When omitted,
buildings are detected from the point cloud.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firesmart_lidar import pipeline as P          # noqa: E402
from firesmart_lidar import export as E            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", required=True, type=Path)
    ap.add_argument("--out", default="output", type=Path)
    ap.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml", type=Path)
    ap.add_argument("--footprints", type=Path, help="optional GeoJSON of building polygons (WGS84)")
    ap.add_argument("--name", default="", help="area name shown on the map")
    a = ap.parse_args()
    t0 = time.time()
    cfg = P.load_config(a.config)
    a.out.mkdir(parents=True, exist_ok=True)

    pts, crs = P.read_points(a.las, cfg)
    cell = cfg["raster"]["cell_m"]
    grid = P.Grid(float(np.floor(pts["x"].min())), float(np.floor(pts["y"].min())), cell,
                  int(np.ceil((pts["x"].max() - np.floor(pts["x"].min())) / cell)),
                  int(np.ceil((pts["y"].max() - np.floor(pts["y"].min())) / cell)))
    print(f"read {len(pts['x']):,} points  crs={crs.to_epsg()}  grid={grid.nx}x{grid.ny}")

    dtm, nonground = P.build_dtm(pts, grid, cfg["classes"]["ground"])
    hag = P.height_above_ground(pts, grid, dtm)

    if a.footprints:
        from pyproj import Transformer
        t = Transformer.from_crs(4326, crs, always_xy=True)
        footprints = P.footprints_from_file(a.footprints, lambda x, y, z=None: t.transform(x, y))
        method = "footprint file"
    else:
        roof, method = P.detect_roof_cells(pts, hag, nonground, grid, cfg)
        footprints = P.footprints_from_mask(roof, grid, cfg)
    print(f"buildings: {len(footprints)} (via {method})")

    import shapely
    GX, GY = grid.centers()
    bld_mask = np.zeros((grid.ny, grid.nx), bool)
    for p in footprints:
        bld_mask |= shapely.contains_xy(p.buffer(0.5), GX, GY)

    chm, ladder_n, overhang = P.vegetation_rasters(pts, hag, nonground, bld_mask, grid, cfg)
    results, rings, outs = P.score_buildings(footprints, chm, ladder_n, overhang, dtm, grid, cfg)
    print(f"scored {len(results)} houses, {len(outs)} outbuildings")

    E.write_vectors(results, rings, outs, crs, a.out)
    meta = E.write_rasters(dtm, chm, ladder_n, bld_mask, grid, crs, cfg, a.out)
    summary = {
        "source": a.las.name, "area_name": a.name, "crs": f"EPSG:{crs.to_epsg()}",
        "points": int(len(pts["x"])), "building_method": method,
        "houses": len(results), "outbuildings": len(outs),
        "ratings": {r: sum(m["rating"] == r for m in results) for r in cfg["scoring"]["ratings"]},
        "config": cfg, "raster": meta, "runtime_s": round(time.time() - t0, 1),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    for m in results:
        print(f"  {m['property_id']}  z1 canopy {m['zone1_canopy_pct']:5.1f}% ladder {m['zone1_ladder_pct']:5.1f}%"
              f" | z2 canopy {m['zone2_canopy_pct']:5.1f}% ladder {m['zone2_ladder_pct']:5.1f}%"
              f"  -> {m['overall_score']:3d} {m['rating']}  {', '.join(m['flags'])}")
    print(f"done in {summary['runtime_s']} s -> {a.out}/")


if __name__ == "__main__":
    main()
