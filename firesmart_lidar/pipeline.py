"""
FireSmart Home Ignition Zone scoring from an airborne LiDAR point cloud.

Steps
  1. read LAS/LAZ, drop noise
  2. ground-only DTM -> height above ground (HAG) for every return
  3. building footprints: class 6 if the tile has it, otherwise roof detection from
     geometry (smooth, single-return surfaces >= 2.5 m), or a footprint file you supply
  4. vegetation rasters: canopy height model and ladder-fuel return counts (1-4 m band)
  5. per building: FireSmart rings (0-1.5, 1.5-10, 10-30 m from the wall), canopy %,
     ladder-fuel %, branches over roof, outbuildings within 10 m, slope
  6. config-driven scores and ratings, plus plain-language recommendations
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import shapely
import shapely.ops
import yaml
from pyproj import CRS
from scipy import ndimage as ndi
from shapely.geometry import box, shape


# --------------------------------------------------------------------------- data
@dataclass
class Grid:
    xmin: float
    ymin: float
    cell: float
    nx: int
    ny: int

    def rc(self, x, y):
        c = np.clip(((x - self.xmin) / self.cell).astype(np.int64), 0, self.nx - 1)
        r = np.clip(((y - self.ymin) / self.cell).astype(np.int64), 0, self.ny - 1)
        return r, c

    def centers(self):
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.ymin + (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_points(path: Path, cfg: dict):
    las = laspy.read(path)
    cls = np.asarray(las.classification)
    keep = ~np.isin(cls, cfg["classes"]["noise"])
    crs = None
    if cfg["crs"].get("prefer_header_crs", True):
        try:
            crs = las.header.parse_crs()
        except Exception:
            crs = None
    if crs is not None and crs.is_compound:
        crs = crs.sub_crs_list[0]          # e.g. UTM 16N + CGVD2013 -> horizontal part only
    if crs is None:
        crs = CRS.from_epsg(cfg["crs"]["source_epsg"])
    if crs.to_epsg() is None:              # header WKT that matches a known EPSG code loosely
        guess = crs.to_epsg(min_confidence=50)
        crs = CRS.from_epsg(guess) if guess else crs
    pts = {
        "x": np.asarray(las.x)[keep],
        "y": np.asarray(las.y)[keep],
        "z": np.asarray(las.z)[keep],
        "cls": cls[keep],
        "rn": np.asarray(las.return_number)[keep],
        "nr": np.asarray(las.number_of_returns)[keep],
    }
    return pts, crs


# --------------------------------------------------------------------------- terrain
def build_dtm(pts, grid: Grid, ground_classes):
    g = np.isin(pts["cls"], ground_classes)
    r, c = grid.rc(pts["x"][g], pts["y"][g])
    idx = r * grid.nx + c
    n = grid.nx * grid.ny
    s = np.bincount(idx, weights=pts["z"][g], minlength=n)
    k = np.bincount(idx, minlength=n)
    dtm = np.where(k > 0, s / np.maximum(k, 1), np.nan).reshape(grid.ny, grid.nx)
    holes = np.isnan(dtm)
    if holes.any():
        _, (ri, ci) = ndi.distance_transform_edt(holes, return_indices=True)
        dtm = dtm[ri, ci]
    dtm = ndi.gaussian_filter(dtm, 0.7)
    return dtm, ~g


def height_above_ground(pts, grid: Grid, dtm):
    rr = (pts["y"] - grid.ymin) / grid.cell - 0.5
    cc = (pts["x"] - grid.xmin) / grid.cell - 0.5
    ground_z = ndi.map_coordinates(dtm, [rr, cc], order=1, mode="nearest")
    return pts["z"] - ground_z


# --------------------------------------------------------------------------- buildings
def detect_roof_cells(pts, hag, nonground, grid: Grid, cfg) -> np.ndarray:
    b = cfg["buildings"]
    n = grid.nx * grid.ny
    bclasses = cfg["classes"]["building"]
    labelled = np.isin(pts["cls"], bclasses)
    if labelled.sum() > 100:                                     # tile has building class
        r, c = grid.rc(pts["x"][labelled], pts["y"][labelled])
        roof = np.bincount(r * grid.nx + c, minlength=n) > 0
        return roof.reshape(grid.ny, grid.nx), "class"

    cand = nonground & (hag > b["min_roof_height_m"])
    r, c = grid.rc(pts["x"][cand], pts["y"][cand])
    idx = r * grid.nx + c
    z = hag[cand]
    cnt = np.bincount(idx, minlength=n)
    s1 = np.bincount(idx, weights=z, minlength=n)
    s2 = np.bincount(idx, weights=z * z, minlength=n)
    multi = np.bincount(idx, weights=(pts["nr"][cand] > 1).astype(float), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0))
        mfrac = multi / cnt
    roof = (cnt >= b["min_points_per_cell"]) & (std <= b["max_cell_z_std_m"]) & \
           (mfrac <= b["max_multi_return_frac"])
    return roof.reshape(grid.ny, grid.nx), "geometry"


def footprints_from_mask(mask, grid: Grid, cfg):
    b = cfg["buildings"]
    mask = ndi.binary_closing(mask, np.ones((3, 3)), iterations=1)
    mask = ndi.binary_opening(mask, np.ones((2, 2)))
    mask = ndi.binary_fill_holes(mask)
    lab, nlab = ndi.label(mask)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        comp = lab[sl] == i
        area = comp.sum() * grid.cell ** 2
        if area < b["min_area_m2"]:
            continue
        rr, cc = np.nonzero(comp)
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        x0 = grid.xmin + cc * grid.cell
        y0 = grid.ymin + rr * grid.cell
        poly = shapely.union_all(shapely.box(x0, y0, x0 + grid.cell, y0 + grid.cell))
        hull = poly.convex_hull
        if poly.area / hull.area < b["min_compactness"]:
            continue                                    # likely a dense flat-topped crown
        rect = poly.minimum_rotated_rectangle
        if poly.area / rect.area > 0.8:
            poly = rect
        else:
            poly = poly.simplify(grid.cell * 0.75)
        out.append(poly)
    return out


def footprints_from_file(path: Path, to_tile_crs):
    import json
    with open(path) as f:
        gj = json.load(f)
    polys = [shape(ft["geometry"]) for ft in gj["features"]]
    return [shapely.ops.transform(to_tile_crs, p) for p in polys]


# --------------------------------------------------------------------------- vegetation
def vegetation_rasters(pts, hag, nonground, bld_mask, grid: Grid, cfg):
    v = cfg["vegetation"]
    n = grid.nx * grid.ny
    r, c = grid.rc(pts["x"], pts["y"])
    idx = r * grid.nx + c
    in_bld = bld_mask.ravel()[idx]

    # Branches over the roof: vegetation returns at least 1 m above the highest roof
    # surface in the same cell. Roof surface = building-class points when the tile has them,
    # otherwise single returns (roofs give one return, foliage usually several).
    is_bclass = np.isin(pts["cls"], cfg["classes"]["building"])
    roof_pts = in_bld & nonground & (hag > 1.5) & (is_bclass if is_bclass.any() else (pts["nr"] == 1))
    roof_top = np.full(n, -np.inf)
    np.maximum.at(roof_top, idx[roof_pts], hag[roof_pts])
    cand = in_bld & nonground & ~is_bclass & np.isfinite(roof_top[idx]) & (hag > roof_top[idx] + 1.0)
    if not is_bclass.any():
        cand &= pts["nr"] > 1
    overhang_cells = (np.bincount(idx[cand], minlength=n) >= v["ladder_min_returns"]).reshape(grid.ny, grid.nx)

    not_veg = np.isin(pts["cls"], cfg["classes"].get("not_vegetation", []) + cfg["classes"]["building"])
    veg = nonground & ~in_bld & ~not_veg & (hag > 0.2) & (hag < 45)
    chm = np.zeros(n)
    np.maximum.at(chm, idx[veg], hag[veg])
    lo, hi = v["ladder_band_m"]
    band = veg & (hag >= lo) & (hag <= hi)
    ladder_n = np.bincount(idx[band], minlength=n)
    return (chm.reshape(grid.ny, grid.nx),
            ladder_n.reshape(grid.ny, grid.nx),
            overhang_cells)


# --------------------------------------------------------------------------- scoring
def ramp(v, ok, max):  # noqa: A002 (matches config key)
    mx = max
    return float(np.clip((v - ok) / (mx - ok), 0, 1) * 100)


def rating(score, cfg):
    for name, ub in cfg["scoring"]["ratings"].items():
        if score <= ub:
            return name
    return list(cfg["scoring"]["ratings"])[-1]


def recommendations(m):
    rec = []
    if m["zone1_ladder_pct"] > 10:
        rec.append("Within 10 m: remove shrubs and young conifers, and prune tree branches up to 2 m.")
    if m["zone1_canopy_pct"] > 20:
        rec.append("Within 10 m: thin trees so crowns don't touch; favour leafy deciduous trees over conifers.")
    if m["zone0_veg_pct"] > 10:
        rec.append("Keep the first 1.5 m around the house non-combustible (gravel, no plants or mulch).")
    if m["branches_over_roof_pct"] > 5:
        rec.append("Trim branches that hang over the roof, and keep gutters clear of needles.")
    if m["outbuilding_within_10m"]:
        rec.append(f"Outbuilding {m['nearest_outbuilding_m']} m from the house: FireSmart it too, or keep it 10 m+ away.")
    if m["zone2_canopy_pct"] > 40:
        rec.append("10-30 m: thin to roughly 3 m between tree crowns.")
    if m["zone2_ladder_pct"] > 20:
        rec.append("10-30 m: clear ladder fuels (shrubs, low branches, regen) under the canopy.")
    if not rec:
        rec.append("Vegetation around the house looks well managed. An on-site assessment can check the house itself.")
    return rec


def score_buildings(footprints, chm, ladder_n, overhang_cells, dtm, grid: Grid, cfg):
    v, z, sc = cfg["vegetation"], cfg["zones_m"], cfg["scoring"]
    GX, GY = grid.centers()
    all_bld = np.zeros((grid.ny, grid.nx), bool)
    for p in footprints:
        mnx, mny, mxx, mxy = p.bounds
        r0, c0 = grid.rc(np.array([mnx]), np.array([mny]))
        r1, c1 = grid.rc(np.array([mxx]), np.array([mxy]))
        sub = (slice(r0[0], r1[0] + 1), slice(c0[0], c1[0] + 1))
        all_bld[sub] |= shapely.contains_xy(p, GX[sub], GY[sub])

    gy, gx = np.gradient(dtm, grid.cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    primaries = [p for p in footprints if p.area >= cfg["buildings"]["primary_min_area_m2"]]
    outs = [p for p in footprints if p.area < cfg["buildings"]["primary_min_area_m2"]]

    results, rings_out = [], []
    canopy = chm >= v["canopy_min_height_m"]
    ladder = ladder_n >= v["ladder_min_returns"]
    for k, p in enumerate(sorted(primaries, key=lambda q: (-q.centroid.y, q.centroid.x))):
        pid = f"P{k + 1:02d}"
        mnx, mny, mxx, mxy = p.buffer(z["zone2"][1] + 2).bounds
        r0, c0 = grid.rc(np.array([mnx]), np.array([mny]))
        r1, c1 = grid.rc(np.array([mxx]), np.array([mxy]))
        sub = (slice(r0[0], r1[0] + 1), slice(c0[0], c1[0] + 1))
        d = shapely.distance(p, shapely.points(GX[sub], GY[sub]))
        free = ~all_bld[sub]
        m = {"property_id": pid, "footprint_m2": round(p.area, 1)}
        for zn in ("zone0", "zone1", "zone2"):
            a, b_ = z[zn]
            ring = (d > a) & (d <= b_) & free
            if a == 0:
                ring = (d > 0) & (d <= b_) & free
            ncell = max(int(ring.sum()), 1)
            if zn == "zone0":
                m["zone0_veg_pct"] = round(100 * float(((chm[sub] > 0.5) & ring).sum()) / ncell, 1)
                continue
            m[f"{zn}_canopy_pct"] = round(100 * float((canopy[sub] & ring).sum()) / ncell, 1)
            m[f"{zn}_ladder_pct"] = round(100 * float((ladder[sub] & ring).sum()) / ncell, 1)
            m[f"{zn}_mean_canopy_ht_m"] = round(float(chm[sub][ring & canopy[sub]].mean()) if (ring & canopy[sub]).any() else 0.0, 1)
            if zn == "zone2":
                m["zone2_mean_slope_deg"] = round(float(slope[sub][ring].mean()), 1)
            ring_poly = p.buffer(b_).difference(p.buffer(a)) if a > 0 else p.buffer(b_).difference(p)
            rings_out.append({"property_id": pid, "zone": zn, "geom": ring_poly})
        rings_out.append({"property_id": pid, "zone": "zone0",
                          "geom": p.buffer(z["zone0"][1]).difference(p)})

        # how much of the 30 m ring the tile actually covers (houses near a tile edge)
        tile = box(grid.xmin, grid.ymin, grid.xmin + grid.nx * grid.cell, grid.ymin + grid.ny * grid.cell)
        z2 = p.buffer(z["zone2"][1]).difference(p)
        m["zone2_coverage_pct"] = round(100 * z2.intersection(tile).area / z2.area, 1)

        fp_cells = shapely.contains_xy(p, GX[sub], GY[sub])
        m["branches_over_roof_pct"] = round(100 * float((overhang_cells[sub] & fp_cells).sum()) / max(fp_cells.sum(), 1), 1)

        dists = [round(p.distance(o), 1) for o in outs if p.distance(o) <= z["zone2"][1]]
        m["nearest_outbuilding_m"] = min(dists) if dists else None
        m["outbuilding_within_10m"] = bool(dists and min(dists) <= z["zone1"][1])

        for zn in ("zone1", "zone2"):
            s = sc[zn]
            cs = ramp(m[f"{zn}_canopy_pct"], **s["canopy_pct"])
            ls = ramp(m[f"{zn}_ladder_pct"], **s["ladder_pct"])
            m[f"{zn}_canopy_score"] = round(cs)
            m[f"{zn}_ladder_score"] = round(ls)
            m[f"{zn}_score"] = round(s["weights"]["canopy"] * cs + s["weights"]["ladder"] * ls)
        overall = sc["overall_weights"]["zone1"] * m["zone1_score"] + sc["overall_weights"]["zone2"] * m["zone2_score"]
        bumps = sc["bumps"]
        flags = []
        if m["branches_over_roof_pct"] > 5:
            overall += bumps["branches_over_roof_pct_gt_5"]; flags.append("branches over roof")
        if m["outbuilding_within_10m"]:
            overall += bumps["outbuilding_within_10m"]; flags.append("outbuilding within 10 m")
        if m["zone2_mean_slope_deg"] > 15:
            overall += bumps["zone2_mean_slope_deg_gt_15"]; flags.append("steep slope")
        if m["zone2_coverage_pct"] < 95:
            flags.append(f"edge of tile: only {m['zone2_coverage_pct']:.0f}% of the 30 m ring measured")
        m["overall_score"] = round(min(overall, 100))
        m["rating"] = rating(m["overall_score"], cfg)
        m["zone1_rating"] = rating(m["zone1_score"], cfg)
        m["zone2_rating"] = rating(m["zone2_score"], cfg)
        m["flags"] = flags
        m["recommendations"] = recommendations(m)
        m["_geom"] = p
        results.append(m)
    return results, rings_out, outs
