#!/usr/bin/env python3
"""Download the official BIRD training database archive.

The archive is intentionally not unpacked automatically. This keeps the script
predictable and lets users inspect disk requirements before extraction.

A multi-gigabyte download over a long connection is the ordinary case here, so a
dropped connection is the ordinary failure. Writing straight to the destination
would leave a truncated archive sitting where a complete one belongs, and a
partially extracted archive is worse than a failed one: it yields a subset of the
databases, so every later split and metric silently describes less data than it
claims to. The download therefore goes to a temporary file, is checked for
completeness, and only then takes the destination name.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
CHUNK_SIZE = 8 * 1024 * 1024
TIMEOUT_SECONDS = 60.0


def _checked_url(url: str) -> str:
    """Reject anything that is not an HTTP(S) URL.

    ``urlopen`` also speaks ``file:`` and ``ftp:``. Neither is meaningful for a
    published benchmark archive, and accepting ``file:`` turns a download into a
    silent local copy.
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Refusing a non-HTTP(S) URL: {url!r}")
    return url


def download(
    url: str,
    output: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    sha256: str | None = None,
    timeout: float = TIMEOUT_SECONDS,
    quiet: bool = False,
) -> Path:
    """Stream a remote file to disk, leaving nothing behind on failure.

    Completeness is checked against ``Content-Length`` when the server sends it,
    which catches a truncated transfer without needing a published digest. Pass
    ``sha256`` to verify the contents as well.
    """
    _checked_url(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    digest = hashlib.sha256()
    written = 0

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            expected = int(response.headers.get("Content-Length", "0"))
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if expected and not quiet:
                        print(f"\r{written / expected:6.1%}", end="", flush=True)
        if not quiet:
            print()

        if expected and written != expected:
            raise OSError(
                f"Truncated download: got {written} bytes, server declared {expected}. "
                "Nothing was written to the destination; re-run to retry."
            )
        if sha256 is not None and digest.hexdigest() != sha256.lower():
            raise ValueError(
                f"Checksum mismatch for {url}\n"
                f"  expected {sha256.lower()}\n"
                f"  received {digest.hexdigest()}"
            )
        partial.replace(output)
    except BaseException:
        # Includes KeyboardInterrupt: an interrupted multi-gigabyte download must
        # not leave a plausible-looking archive behind.
        partial.unlink(missing_ok=True)
        raise

    if not quiet:
        print(f"Saved {output} ({written:,} bytes, sha256 {digest.hexdigest()})")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/bird_train.zip"))
    parser.add_argument("--url", default=URL)
    parser.add_argument(
        "--sha256",
        help=(
            "Expected SHA-256 of the archive. The project does not hard-code one "
            "because the benchmark publishes releases independently; take it from "
            "the BIRD download page and pass it here to verify the contents."
        ),
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    download(args.url, args.output, sha256=args.sha256, timeout=args.timeout)


if __name__ == "__main__":
    main()
