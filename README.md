# FireSmart LiDAR neighbourhood risk map

Scores every house in an airborne LiDAR tile against the FireSmart Home Ignition Zone and builds a
self-contained interactive map. Each house gets these measures:

| Measure | What it is |
|---|---|
| **Zone 1 canopy %** | Share of the 1.5–10 m ring covered by vegetation ≥ 3 m tall |
| **Zone 1 ladder fuel %** | Share of the 1.5–10 m ring with LiDAR returns between 1 and 4 m (shrubs, regen, low branches) |
| **Zone 2 canopy % / ladder %** | Same, for the 10–30 m ring |
| Extras | Vegetation in the first 1.5 m, branches over the roof, outbuildings within 10 m, slope |

Scores and ratings (Low / Moderate / High / Extreme) are driven entirely by `config.yaml`.
You can tune the thresholds with a local FireSmart rep without touching code.

## Quick start

```bash
pip install -r requirements.txt

# 1. demo tile (synthetic, ~4.3 M points, rural road near Searchmont)
python scripts/make_synthetic_tile.py --out data/synthetic_searchmont.laz

# 2. score
python scripts/run_pipeline.py --las data/synthetic_searchmont.laz --out output

# 3. check against the synthetic truth (synthetic tiles only)
python scripts/validate_synthetic.py data/synthetic_searchmont.truth.json output/properties.geojson

# 4. bundle the map (single HTML file, open in any browser)
python scripts/build_map.py --out output
```

A live demo on synthetic data is in `docs/index.html` (enable GitHub Pages on `/docs` to publish it).

Outputs in `output/`: `risk_map.html`, `properties.geojson`, `zones.geojson` (WGS84),
`properties.csv`, `canopy.png` / `ladder.png` overlays, `summary.json`.

## Several tiles at once

```bash
python scripts/run_mosaic.py --tiles "path/to/tiles/*.laz" --out output_mosaic --config config_ssm2024.yaml --name Searchmont
python scripts/build_map.py --out output_mosaic --tag "2024 LiDAR" \
    --attribution "LiDAR: NRCan CanElevation / Ontario, 2024 · Open Government Licence – Canada"
```

Tiles are streamed one at a time, so memory stays near one tile. Houses near an inner tile edge get their full
30 m ring from the neighbour. Nine 2024 Sault Ste Marie tiles (about 200 M points) take ~5 minutes.

## Getting Searchmont tiles

The 2024 Sault Ste Marie lift is public on NRCan's CanElevation bucket (no ordering needed):

```
https://canelevation-lidar-point-clouds.s3.ca-central-1.amazonaws.com/pointclouds_nuagespoints/ON/Sault_SteMarie_UTM16_2024/1km16EEE0NNNN02024LSAULTSTEMARIE_CPC.copc.laz
```

`EEE` is the easting in km and `NNNN` the northing in km (UTM 16N). Searchmont's 3 × 3 block is EEE 724–726,
NNNN 5183–5185. Use `config_ssm2024.yaml` for these tiles (~20 pts/m², buildings pre-classified).

## Running it on real Ontario data

1. Find your area in the **Ontario Classified Point Cloud (Lidar-Derived)** on Ontario GeoHub
   (https://geohub.lio.gov.on.ca/datasets/mnrf::ontario-classified-point-cloud-lidar-derived).
   Use the tile index and project extents to find the tile; the Sault Ste Marie 2023–24 and
   Northeastern Ontario 2024 lifts are the likely ones for Searchmont / Algoma.
2. Download the `.laz` tile and run:
   ```bash
   python scripts/run_pipeline.py --las path/to/tile.laz --out output_real
   python scripts/build_map.py --out output_real --tag "Ontario GeoHub LiDAR" \
       --attribution "Contains information licensed under the Open Government Licence – Ontario"
   ```
3. Notes for real tiles:
   * **CRS** is read from the file header. If the header has none, set `crs.source_epsg` in
     `config.yaml`: 3160 for UTM 16N (west of 84°W), 2958 for UTM 17N.
   * **Buildings**: many Ontario deliveries classify only ground, so roofs are detected from
     geometry. Houses tucked under canopy can be missed or misshapen. For best results pass a
     footprint file with `--footprints buildings.geojson`, for example from Ontario Building
     Footprints or Microsoft's Canadian Building Footprints.
   * **Point density**: `vegetation.ladder_min_returns: 1` suits about 8 pts/m². Raise it for
     denser tiles, or ladder fuel will be over-counted.
   * **Large tiles**: the pipeline loads the whole tile into memory, which is fine for a 1 km tile.
     For bigger areas, clip first, e.g. `pdal translate in.laz out.laz crop --filters.crop.bounds=...`.

## How the demo checks out

On the synthetic tile, with known truth for every lot:

* 14 / 14 houses found from geometry alone, footprint centroids within ~1 m; 7 outbuildings found.
* Rank agreement between how well a yard is managed and its score: Spearman −0.98.
* Canopy % reads ~7 points high. Crown edges spill into 1 m cells, so it errs on the side of caution.
* Ladder fuel % reads ~7–8 points low. Low branches inside dense crowns are partly hidden from the air.

Full numbers are in `output/validation_report.txt`.

## Limits (say this out loud at a Prep Day event)

* LiDAR sees vegetation and terrain, not the house. Vents, gutters, decks, siding and roofing are
  where embers get in, and only an on-site FireSmart assessment checks them.
* It can't tell conifer from deciduous on its own. Adding leaf-off imagery or intensity is a
  natural next step.
* Tiles are snapshots. Yard work since the flight won't show.
* Privacy: share property-level results with the owner. Use rings or block-level summaries for
  anything public, and make it opt-in.

## Project layout

```
config.yaml                    thresholds, zones, scoring (edit this)
firesmart_lidar/pipeline.py    DTM, height above ground, building detection, zone metrics, scores
firesmart_lidar/export.py      GeoJSON/CSV writers, web-mercator raster overlays
scripts/make_synthetic_tile.py synthetic Ontario-style LAS 1.4 tile + truth JSON
scripts/run_pipeline.py        CLI
scripts/validate_synthetic.py  compare output to truth
scripts/build_map.py           bundle the single-file HTML map
web/template.html              map page (Leaflet)
```
