#!/usr/bin/env python3
"""Reliably resume one public Google Drive file through an HTTP proxy.

gdown uses an atomic temporary file, but a connection reset can leave that
temporary file behind and subsequent invocations may skip it.  This helper
keeps an explicit ``.part`` file and resumes with HTTP Range requests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests
from gdown.download import _get_session, get_url_from_gdrive_confirmation


def resolve_download(file_id: str, proxy: str):
    """Return a session and confirmed Google Drive URL for a public file."""
    session, _ = _get_session(
        proxy=proxy,
        use_cookies=True,
        user_agent=None,
        return_cookies_file=True,
    )
    url = f"https://drive.google.com/uc?id={file_id}"
    for _ in range(8):
        # Google Drive may show the quota HTML page for a normal GET while still
        # serving the confirmed URL to resumable byte-range requests.  Ask for
        # one byte here so the confirmation flow stays compatible with the
        # range-based transfer below.
        response = session.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(20, 60),
        )
        content_type = response.headers.get("Content-Type", "")
        if "Content-Disposition" in response.headers and not content_type.startswith("text/html"):
            response.close()
            return session, url
        if not content_type.startswith("text/html"):
            response.raise_for_status()
            raise RuntimeError(f"Google Drive returned no file attachment: {response.headers}")
        html = response.text
        response.close()
        if "Quota exceeded" in html or "download quota" in html.lower():
            raise RuntimeError("Google Drive download quota exceeded for this file")
        url = get_url_from_gdrive_confirmation(html)
        # The generated drive.usercontent URL is single-use in practice: a
        # probe can consume it and make the following request fall back to a
        # non-range response.  Return it for the actual first data request.
        return session, url
    raise RuntimeError("too many Google Drive confirmation redirects")


def content_range_total(value: str) -> int:
    matched = re.fullmatch(r"bytes \d+-\d+/(\d+)", value)
    if not matched:
        raise RuntimeError(f"unexpected Content-Range header: {value!r}")
    return int(matched.group(1))


def download(file_id: str, output: Path, proxy: str, retries: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    expected_total: int | None = None

    for attempt in range(1, retries + 1):
        session, url = resolve_download(file_id, proxy)
        # Always use Range, including at offset zero.  Besides enabling
        # recovery, this avoids Google Drive's unreliable whole-file response.
        headers = {"Range": f"bytes={offset}-"}
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(20, 60))
            if response.status_code != 206:
                raise RuntimeError(
                    f"server did not honor range at byte {offset}: HTTP {response.status_code}"
                )
            expected_total = content_range_total(response.headers.get("Content-Range", ""))

            with partial.open("ab") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
                        offset += len(block)
                        if expected_total:
                            print(
                                f"attempt={attempt} bytes={offset}/{expected_total} "
                                f"({offset / expected_total:.1%})",
                                flush=True,
                            )
                stream.flush()
        except (requests.RequestException, OSError) as error:
            print(f"attempt={attempt} interrupted at byte {offset}: {error}", file=sys.stderr, flush=True)
            continue
        finally:
            try:
                response.close()
            except UnboundLocalError:
                pass
            session.close()

        if expected_total is not None and offset == expected_total:
            partial.replace(output)
            print(f"complete: {output} ({offset} bytes)", flush=True)
            return
        print(f"attempt={attempt} ended early at byte {offset}; reconnecting", file=sys.stderr, flush=True)

    raise RuntimeError(f"download did not complete after {retries} attempts; partial={partial}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()
    download(args.file_id, args.output, args.proxy, args.retries)


if __name__ == "__main__":
    main()
