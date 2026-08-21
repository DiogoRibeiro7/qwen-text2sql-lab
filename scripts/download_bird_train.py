#!/usr/bin/env python3
"""Download the official BIRD training database archive.

The archive is intentionally not unpacked automatically. This keeps the script
predictable and lets users inspect disk requirements before extraction.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"


def download(url: str, output: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Stream a remote file to disk without holding it in memory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0"))
        written = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if total:
                print(f"\r{written / total:6.1%}", end="", flush=True)
    print(f"\nSaved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/bird_train.zip"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    download(args.url, args.output)


if __name__ == "__main__":
    main()
