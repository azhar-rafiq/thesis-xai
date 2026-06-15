#!/usr/bin/env python3
"""download Seg-CQ500 from Zenodo and extract to RDS."""

import time
import zipfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/8063221/files/Seg-CQ500.zip?download=1"
OUT_DIR = Path("/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/seg-cq500")
ZIP_PATH = OUT_DIR / "Seg-CQ500.zip"


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        print(f"zip already present: {ZIP_PATH}")
        return

    tmp = ZIP_PATH.with_suffix(".tmp")
    start = tmp.stat().st_size if tmp.exists() else 0

    headers = {}
    if start:
        headers["Range"] = f"bytes={start}-"
        print(f"resuming from {format_bytes(start)}")

    req = urllib.request.Request(URL, headers=headers)
    print(f"downloading -> {ZIP_PATH}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0)) + start
        mode = "ab" if start else "wb"
        downloaded = start
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1 << 17)  # 128 KB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - t0, 0.001)
                speed = format_bytes(int((downloaded - start) / elapsed)) + "/s"
                pct = f"{100*downloaded/total:.1f}%" if total else "?%"
                print(f"  {pct}  {format_bytes(downloaded)}/{format_bytes(total)}  {speed}    ", end="\r", flush=True)

    tmp.rename(ZIP_PATH)
    print(f"\ndownload complete: {format_bytes(ZIP_PATH.stat().st_size)}")


def extract():
    print(f"extracting -> {OUT_DIR}/")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = zf.namelist()
        for i, member in enumerate(members, 1):
            dest = OUT_DIR / member
            if not dest.exists():
                zf.extract(member, OUT_DIR)
            print(f"  [{i}/{len(members)}] {member}    ", end="\r", flush=True)
    print(f"\nextraction complete.")


if __name__ == "__main__":
    download()
    extract()
    ZIP_PATH.unlink()
    print("zip removed.")
