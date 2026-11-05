"""
Download LiDAR tiles you don't have yet from NRCan's public CanElevation bucket.

Give it the tile list you exported from the Ontario GeoHub tile index (any text file that
contains tile names like 1km167250518402024LSAULTSTEMARIE works), and a folder. Tiles
already in the folder are skipped; interrupted downloads resume cleanly on the next run.

    python scripts/fetch_tiles.py --list tiles_searchmont_corridor.txt --dir "D:/lidar"

Standard library only, so it runs before anything is installed.
"""
import argparse
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = ("https://canelevation-lidar-point-clouds.s3.ca-central-1.amazonaws.com/"
        "pointclouds_nuagespoints/ON/Sault_SteMarie_UTM16_2024/")
NAME_RE = re.compile(r"(1km\d{2}\d{3}0\d{4}0\d{4}L[A-Z]+)")
lock = threading.Lock()


def have(folder: Path, tile: str) -> bool:
    return any(p.stat().st_size > 0 for p in folder.glob(f"{tile}*.laz"))


def download(url: str, dest: Path, tries: int = 4) -> int:
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                size = int(r.headers["Content-Length"])
            done = part.stat().st_size if part.exists() else 0
            if done > size:
                part.unlink(); done = 0
            if done < size:
                req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if done else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            if part.stat().st_size != size:
                raise IOError(f"size mismatch {part.stat().st_size} != {size}")
            part.replace(dest)
            return size
        except Exception as e:                     # network hiccup: wait and resume
            if attempt == tries:
                raise
            with lock:
                print(f"    retry {attempt} for {dest.name}: {e}", flush=True)
            time.sleep(5 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, type=Path, help="text/CSV exported from the GeoHub tile index")
    ap.add_argument("--dir", required=True, type=Path, help="folder that holds (and receives) the tiles")
    ap.add_argument("--base", default=BASE, help="bucket folder for the project")
    ap.add_argument("--threads", default=3, type=int)
    ap.add_argument("--dry-run", action="store_true", help="only report what would be downloaded")
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(set(NAME_RE.findall(a.list.read_text(errors="ignore"))))
    todo = [t for t in tiles if not have(a.dir, t)]
    print(f"{len(tiles)} tiles in list, {len(tiles) - len(todo)} already here, {len(todo)} to download")
    if not todo or a.dry_run:
        return
    t0, got, n = time.time(), 0, 0
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(download, a.base + f"{t}_CPC.copc.laz", a.dir / f"{t}_CPC.copc.laz"): t for t in todo}
        failed = []
        for f in as_completed(futs):
            n += 1
            try:
                got += f.result()
                rate = got / 1e6 / max(time.time() - t0, 1)
                with lock:
                    print(f"  [{n:3d}/{len(todo)}] {futs[f]}  {got / 1e9:5.1f} GB  {rate:5.1f} MB/s", flush=True)
            except Exception as e:
                failed.append(futs[f])
                print(f"  [{n:3d}/{len(todo)}] FAILED {futs[f]}: {e}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} tiles failed; run the same command again to resume them")
    print(f"done: {got / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
