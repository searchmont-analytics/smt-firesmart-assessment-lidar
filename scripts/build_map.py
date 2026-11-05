"""Bundle pipeline output into one self-contained HTML map (no server needed)."""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def b64(path):
    mime = "image/webp" if str(path).endswith(".webp") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output", type=Path, help="pipeline output folder")
    ap.add_argument("--leaflet-css", type=Path, default=ROOT / "web" / "leaflet.css")
    ap.add_argument("--html", type=Path, default=None)
    ap.add_argument("--title", default="Searchmont Ignition Zone Map")
    ap.add_argument("--h1", default="Searchmont neighbourhood wildfire map")
    ap.add_argument("--tag", default="Synthetic demo data")
    ap.add_argument("--intro", default=(
        "Every house scored on tree canopy and ladder fuel within 10 m and 30 m, measured from an "
        "airborne LiDAR point cloud. This demo runs on a simulated tile laid out like a rural road near "
        "Searchmont; the same code runs on Ontario GeoHub tiles."))
    ap.add_argument("--attribution", default="Synthetic LiDAR tile · method after FireSmart Canada HIZ")
    a = ap.parse_args()

    summary = json.loads((a.out / "summary.json").read_text())
    data = {
        "properties": json.loads((a.out / "properties.geojson").read_text()),
        "zones": json.loads((a.out / "zones.geojson").read_text()),
        "raster": summary["raster"],
        "scoring": summary["config"]["scoring"],
        "ratings": summary["config"]["scoring"]["ratings"],
        "attribution": a.attribution,
    }
    html = (ROOT / "web" / "template.html").read_text()
    html = (html.replace("/*__LEAFLET_CSS__*/", a.leaflet_css.read_text())
                .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
                .replace("__CANOPY__", b64(a.out / summary["raster"].get("canopy_file", "canopy.png")))
                .replace("__LADDER__", b64(a.out / "ladder.png"))
                .replace("__TITLE__", a.title).replace("__H1__", a.h1)
                .replace("__TAG__", a.tag).replace("__INTRO__", a.intro))
    dest = a.html or (a.out / "risk_map.html")
    dest.write_text(html)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
