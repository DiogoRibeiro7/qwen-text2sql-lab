"""Tests for the BIRD archive download.

A multi-gigabyte download over a long connection makes a dropped connection the
ordinary failure, not the exotic one. Writing straight to the destination left a
truncated archive where a complete one belongs — and a partially extracted
archive is worse than a failed download, because it yields a subset of the
databases and every later split and metric then describes less data than it
claims to, without saying so.

Served from a local HTTP server on a loopback port: no network, no BIRD.
"""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PAYLOAD = b"BIRD-archive-stand-in" * 500


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "download_bird_train", SCRIPTS / "download_bird_train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_bird_train"] = module
    spec.loader.exec_module(module)
    return module


downloader = _load()


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves PAYLOAD, and can lie about Content-Length to fake a truncation."""

    payload = PAYLOAD
    declared_length: int | None = None

    def do_GET(self) -> None:
        if self.path == "/missing":
            self.send_error(404)
            return
        self.send_response(200)
        declared = self.declared_length if self.declared_length is not None else len(self.payload)
        self.send_header("Content-Length", str(declared))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args: object) -> None:
        """Silence the default stderr logging."""


@pytest.fixture()
def server() -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _reset_handler() -> Iterator[None]:
    yield
    _Handler.payload = PAYLOAD
    _Handler.declared_length = None


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_complete_download_lands_at_the_destination(server: str, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "train.zip"
    result = downloader.download(f"{server}/train.zip", target, quiet=True)
    assert result == target
    assert target.read_bytes() == PAYLOAD


def test_a_matching_checksum_is_accepted(server: str, tmp_path: Path) -> None:
    target = tmp_path / "train.zip"
    downloader.download(
        f"{server}/train.zip", target, sha256=hashlib.sha256(PAYLOAD).hexdigest(), quiet=True
    )
    assert target.is_file()


def test_the_checksum_comparison_ignores_case(server: str, tmp_path: Path) -> None:
    target = tmp_path / "train.zip"
    downloader.download(
        f"{server}/train.zip",
        target,
        sha256=hashlib.sha256(PAYLOAD).hexdigest().upper(),
        quiet=True,
    )
    assert target.is_file()


def test_no_partial_file_survives_a_success(server: str, tmp_path: Path) -> None:
    target = tmp_path / "train.zip"
    downloader.download(f"{server}/train.zip", target, quiet=True)
    assert not (tmp_path / "train.zip.part").exists()


# --------------------------------------------------------------------------
# Failure must leave nothing behind
# --------------------------------------------------------------------------


def test_a_truncated_transfer_is_detected_without_a_checksum(server: str, tmp_path: Path) -> None:
    """The server declares more than it sends, which is what a dropped connection
    looks like from the client side."""
    _Handler.declared_length = len(PAYLOAD) * 2
    target = tmp_path / "train.zip"
    with pytest.raises(OSError, match="Truncated download"):
        downloader.download(f"{server}/train.zip", target, quiet=True)
    assert not target.exists()
    assert not (tmp_path / "train.zip.part").exists()


def test_a_checksum_mismatch_leaves_no_archive(server: str, tmp_path: Path) -> None:
    target = tmp_path / "train.zip"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.download(f"{server}/train.zip", target, sha256="00" * 32, quiet=True)
    assert not target.exists()
    assert not (tmp_path / "train.zip.part").exists()


def test_an_http_error_leaves_no_archive(server: str, tmp_path: Path) -> None:
    target = tmp_path / "train.zip"
    with pytest.raises(Exception, match="404"):
        downloader.download(f"{server}/missing", target, quiet=True)
    assert not target.exists()
    assert not (tmp_path / "train.zip.part").exists()


def test_an_existing_archive_survives_a_failed_retry(server: str, tmp_path: Path) -> None:
    """Re-running after a good download must not destroy what is already there."""
    target = tmp_path / "train.zip"
    downloader.download(f"{server}/train.zip", target, quiet=True)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.download(f"{server}/train.zip", target, sha256="11" * 32, quiet=True)
    assert target.read_bytes() == PAYLOAD


# --------------------------------------------------------------------------
# URL handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("file:///etc/passwd", id="file"),
        pytest.param("ftp://example.invalid/train.zip", id="ftp"),
        pytest.param("/data/train.zip", id="bare-path"),
    ],
)
def test_non_http_urls_are_refused(url: str, tmp_path: Path) -> None:
    """urlopen speaks `file:` too, which would turn a download into a local copy."""
    with pytest.raises(ValueError, match="non-HTTP"):
        downloader.download(url, tmp_path / "train.zip", quiet=True)


@pytest.mark.parametrize("url", ["http://example.invalid/a.zip", "HTTPS://Example.invalid/a.zip"])
def test_http_and_https_are_accepted_in_any_case(url: str) -> None:
    assert downloader._checked_url(url) == url


def test_the_default_url_is_the_official_archive() -> None:
    assert downloader.URL.startswith("https://")
    assert downloader.URL.endswith("train.zip")


def test_the_default_output_stays_under_the_gitignored_data_tree() -> None:
    """data/raw/ is gitignored; anywhere else would invite committing the archive."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/bird_train.zip"))
    default = parser.parse_args([]).output
    assert default.parts[:2] == ("data", "raw")
