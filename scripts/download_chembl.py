#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm

DEFAULT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_37/chembl_37.sdf.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official ChEMBL structure SDF")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/chembl/chembl_37.sdf.gz")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        print(json.dumps({"status": "already_present", "output": str(args.output)}, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    partial.unlink(missing_ok=True)
    started = time.perf_counter()
    request = urllib.request.Request(args.url, headers={"User-Agent": "casmi26/0.5"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as destination:
            expected = int(response.headers.get("Content-Length", 0))
            with tqdm(total=expected or None, unit="B", unit_scale=True, desc="ChEMBL 37") as bar:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
                    bar.update(len(chunk))
        if expected and partial.stat().st_size != expected:
            raise OSError(
                f"Downloaded {partial.stat().st_size} bytes but expected {expected}"
            )
        partial.replace(args.output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "downloaded",
                "url": args.url,
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "runtime_seconds": time.perf_counter() - started,
                "license": "CC BY-SA 3.0",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
