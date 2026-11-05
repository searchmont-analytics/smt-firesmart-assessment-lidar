"""Write results as GeoJSON (WGS84), CSV and web-mercator raster overlays for the map."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import shapely
import shapely.ops
from matplotlib import cm, colors
from matplotlib.image import imsave
from pyproj import Transformer
from scipy import ndimage as ndi


def _to_wgs(crs):
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    return lambda geom: shapely.ops.transform(lambda x, y, z=None: t.transform(x, y), geom)


def _round_geom(g, nd=7):
    return json.loads(json.dumps(shapely.geometry.mapping(g)),
                      parse_float=lambda s: round(float(s), nd))


def write_vectors(results, rings, outbuildings, crs, out: Path):
    to_wgs = _to_wgs(crs)
    feats = []
    for m in results:
        props = {k: v for k, v in m.items() if not k.startswith("_")}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": _round_geom(to_wgs(m["_geom"]))})
    for i, o in enumerate(outbuildings):
        feats.append({"type": "Feature",
                      "properties": {"property_id": f"OB{i + 1:02d}", "outbuilding": True,
                                     "footprint_m2": round(o.area, 1)},
                      "geometry": _round_geom(to_wgs(o))})
    (out / "properties.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    rfeats = [{"type": "Feature", "properties": {"property_id": r["property_id"], "zone": r["zone"]},
               "geometry": _round_geom(to_wgs(r["geom"].simplify(0.3)))} for r in rings]
    (out / "zones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": rfeats}))

    cols = [k for k in results[0] if not k.startswith("_") and k not in ("recommendations",)]
    with open(out / "properties.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["recommendations"])
        w.writeheader()
        for m in results:
            row = {k: m[k] for k in cols}
            row["flags"] = "; ".join(m["flags"])
            row["recommendations"] = " | ".join(m["recommendations"])
            w.writerow(row)


def _hillshade(dtm, cell, az=315, alt=40):
    gy, gx = np.gradient(dtm, cell)
    slope = np.arctan(np.hypot(gx, gy) * 3)
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(90 - alt)
    hs = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def write_rasters(dtm, chm, ladder_n, bld_mask, grid, crs, cfg, out: Path, px_m=0.5, canopy_fmt="png"):
    """Resample to EPSG:3857 so the overlay lines up exactly with the vectors in Leaflet."""
    to_merc = Transformer.from_crs(crs, 3857, always_xy=True)
    back = Transformer.from_crs(3857, crs, always_xy=True)
    xs = [grid.xmin, grid.xmin + grid.nx * grid.cell]
    ys = [grid.ymin, grid.ymin + grid.ny * grid.cell]
    cx, cy = np.meshgrid(xs, ys)
    mx, my = to_merc.transform(cx.ravel(), cy.ravel())
    mx0, mx1, my0, my1 = min(mx), max(mx), min(my), max(my)
    scale = 1 / np.cos(np.radians(46.8))                      # merc metres per ground metre
    step = px_m * scale
    W = int((mx1 - mx0) / step)
    H = int((my1 - my0) / step)
    MX, MY = np.meshgrid(mx0 + (np.arange(W) + 0.5) * step, my1 - (np.arange(H) + 0.5) * step)
    ux, uy = back.transform(MX.ravel(), MY.ravel())
    cc = (np.asarray(ux) - grid.xmin) / grid.cell - 0.5
    rr = (np.asarray(uy) - grid.ymin) / grid.cell - 0.5
    inside = (cc > -0.5) & (cc < grid.nx - 0.5) & (rr > -0.5) & (rr < grid.ny - 0.5)

    def sample(a, order=1):
        v = ndi.map_coordinates(a.astype(float), [rr, cc], order=order, mode="nearest")
        return v.reshape(H, W)

    hs = sample(_hillshade(dtm, grid.cell))
    ch = sample(chm)
    lad = sample((ladder_n >= cfg["vegetation"]["ladder_min_returns"]).astype(float), 0)
    bm = sample(bld_mask.astype(float), 0)
    ins = inside.reshape(H, W)

    # base: hillshaded terrain tinted by canopy height
    ground = np.dstack([0.86 * np.ones_like(hs), 0.83 * np.ones_like(hs), 0.74 * np.ones_like(hs)])
    greens = colors.LinearSegmentedColormap.from_list("g", ["#cfd8a4", "#6f9a54", "#2f5a32", "#183c22"])
    canopy_rgb = greens(np.clip(ch / 22, 0, 1))[..., :3]
    a = np.clip((ch - 0.5) / 3, 0, 1)[..., None]
    rgb = ground * (1 - a) + canopy_rgb * a
    rgb = rgb * (0.55 + 0.45 * hs[..., None])
    rgb[bm > 0.5] = [0.93, 0.93, 0.93]
    rgba = np.dstack([rgb, ins.astype(float)])
    for stale in ("canopy.png", "canopy.webp"):
        (out / stale).unlink(missing_ok=True)
    if canopy_fmt == "webp":                      # ~10x smaller for large mosaics
        from PIL import Image
        Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(
            out / "canopy.webp", "WEBP", quality=82, method=4)
    else:
        imsave(out / "canopy.png", np.clip(rgba, 0, 1))

    lad_rgba = np.zeros((H, W, 4))
    lad_rgba[..., :3] = colors.to_rgb("#e8590c")
    lad_rgba[..., 3] = np.where((lad > 0.5) & ins, 0.75, 0)
    imsave(out / "ladder.png", lad_rgba)

    to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
    (lon0, lat0), (lon1, lat1) = to_ll.transform(mx0, my0), to_ll.transform(mx1, my1)
    meta = {"bounds": [[lat0, lon0], [lat1, lon1]], "px": [W, H], "canopy_file": f"canopy.{canopy_fmt}"}
    (out / "rasters.json").write_text(json.dumps(meta))
    return meta
