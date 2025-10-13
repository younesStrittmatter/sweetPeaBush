# utils/download_data/osf_download.py
from __future__ import annotations

import os
import re
import time
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

_OSF_API = "https://api.osf.io/v2"
_WB_ZIP_ROOT = "https://files.osf.io/v1/resources/{guid}/providers/osfstorage/?zip="
_GUID_RX = re.compile(r"^[a-z0-9]{5}$", re.I)


def download_osf_project(
    *,
    osf_id: str,
    dest_dir: str,
    max_bytes: int = 1_000_000_000,
    request_timeout: float = 30.0,
    max_retries: int = 5,
) -> Dict[str, object]:
    """
    Recursively download all files from OSF Storage for the given project/component GUID.

    Primary: JSON:API walk of osfstorage (stream each file)
    Fallback: WaterButler 'zip whole folder' (UI-style "Download as Zip") to fetch everything at once.

    Env:
      - OSF_TOKEN / OSF_PAT  : bearer token (optional; needed for private projects)
      - SWEETEXTRACT_OSF_ZIP_FALLBACK=1 : force using the zip fallback immediately

    Returns {downloaded_bytes:int, files:List[str], storage_provider:"osfstorage", mode:"api"|"zip_fallback"}
    """
    osf_id = osf_id.strip().lower()
    if not _GUID_RX.match(osf_id):
        raise ValueError(f"Invalid OSF id '{osf_id}' (expect 5 alnum chars)")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # If user wants to force the zip fallback, do so up front.
    if str(os.getenv("SWEETEXTRACT_OSF_ZIP_FALLBACK", "0")).strip().lower() in {"1", "true", "yes"}:
        return _download_zip_fallback(osf_id=osf_id, dest=dest, max_bytes=max_bytes, request_timeout=request_timeout, max_retries=max_retries)

    sess = _make_session(request_timeout=request_timeout, max_retries=max_retries)

    try:
        # ---- API path --------------------------------------------------------
        providers_url = f"{_OSF_API}/nodes/{osf_id}/files/"
        providers = _get_json(sess, providers_url)

        osfstorage_href = None
        for item in providers.get("data", []):
            attrs = item.get("attributes") or {}
            if (attrs.get("provider") or "").lower() == "osfstorage":
                links = item.get("links") or {}
                rel = links.get("related") or {}
                osfstorage_href = rel.get("href") or links.get("self")
                if osfstorage_href:
                    break

        if not osfstorage_href:
            raise RuntimeError(f"No 'osfstorage' provider found for node '{osf_id}'")

        total = 0
        written: List[str] = []

        for node in _walk_folder(sess, osfstorage_href):
            kind = ((node.get("attributes") or {}).get("kind") or "").lower()
            if kind != "file":
                continue

            attrs = node.get("attributes") or {}
            name = attrs.get("name") or "unnamed"
            mat_path = (attrs.get("materialized_path") or f"/{name}").lstrip("/")
            rel_path = _sanitize_rel_path(mat_path)

            dl_url = ((node.get("links") or {}).get("download")) or _infer_download(node)
            if not dl_url:
                continue

            # If server gives us a length, pre-check against budget
            content_len = _head_content_length(sess, dl_url)
            if content_len is not None and total + content_len > max_bytes:
                raise RuntimeError(
                    f"Download budget exceeded: would add {content_len} bytes to {total} with cap {max_bytes}"
                )

            out_path, bytes_written = _stream_download(
                sess, dl_url, dest / rel_path, max_left=max_bytes - total
            )
            total += bytes_written
            written.append(str(Path(out_path).relative_to(dest)))

            if total >= max_bytes:
                raise RuntimeError(f"Download budget reached ({total} >= {max_bytes}). Increase SWEETEXTRACT_MAX_DOWNLOAD_BYTES.")

        # If API path yielded nothing (rare), fall back to zip.
        if total == 0 and not written:
            return _download_zip_fallback(osf_id=osf_id, dest=dest, max_bytes=max_bytes, request_timeout=request_timeout, max_retries=max_retries)

        return {
            "downloaded_bytes": total,
            "files": written,
            "storage_provider": "osfstorage",
            "mode": "api",
        }

    except Exception:
        # Any API traversal failure → try the zip fallback
        return _download_zip_fallback(osf_id=osf_id, dest=dest, max_bytes=max_bytes, request_timeout=request_timeout, max_retries=max_retries)


# ---------------------------- Fallback (ZIP) -----------------------------------

def _download_zip_fallback(
    *,
    osf_id: str,
    dest: Path,
    max_bytes: int,
    request_timeout: float,
    max_retries: int,
) -> Dict[str, object]:
    """
    Download entire OSF Storage as a single zip and extract.

    Uses WaterButler 'zip' flag on the provider root:
      https://files.osf.io/v1/resources/<GUID>/providers/osfstorage/?zip=
    """
    sess = _make_session(request_timeout=request_timeout, max_retries=max_retries)
    zip_url = _WB_ZIP_ROOT.format(guid=osf_id)

    # HEAD to see if server gives us size
    size = _head_content_length(sess, zip_url)
    if size is not None and size > max_bytes:
        raise RuntimeError(f"Zip size {size} exceeds max_bytes {max_bytes}")

    # Stream zip → tmp, enforcing budget
    tmp_zip = dest / ".osfstorage.zip.part"
    final_zip = dest / ".osfstorage.zip"
    if tmp_zip.exists():
        tmp_zip.unlink()

    bytes_written = 0
    with _retry(sess, "GET", zip_url, stream=True, allow_redirects=True) as r:
        with open(tmp_zip, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if bytes_written + len(chunk) > max_bytes:
                    fh.flush()
                    fh.close()
                    tmp_zip.unlink(missing_ok=True)
                    raise RuntimeError("Exceeded max_bytes during zip download")
                fh.write(chunk)
                bytes_written += len(chunk)

    shutil.move(str(tmp_zip), str(final_zip))

    # Safe extract (prevent ZipSlip)
    extracted_files: List[str] = []
    dest_abs = dest.resolve()

    def _safe_dest(p: Path) -> Path:
        ap = p.resolve()
        if not str(ap).startswith(str(dest_abs)):
            raise RuntimeError("Unsafe path in zip (ZipSlip)")
        return ap

    with zipfile.ZipFile(final_zip, "r") as zf:
        for member in zf.infolist():
            # Directories have empty filename in some zips; skip these
            if member.is_dir():
                continue
            out_path = _safe_dest(dest / member.filename)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            # Normalize to dest-relative path
            try:
                extracted_files.append(str(out_path.relative_to(dest)))
            except Exception:
                extracted_files.append(str(out_path))

    # Optional: keep or remove the zip (we keep it hidden by default)
    # final_zip.unlink(missing_ok=True)

    return {
        "downloaded_bytes": bytes_written,
        "files": extracted_files,
        "storage_provider": "osfstorage",
        "mode": "zip_fallback",
    }


# ---------------------------- Internals (API) ----------------------------------

def _make_session(*, request_timeout: float, max_retries: int) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/vnd.api+json",
        "User-Agent": "sweetExtract-osf-downloader/1.1 (+https://github.com/AutoResearch)",
    })
    token = os.getenv("OSF_TOKEN") or os.getenv("OSF_PAT")  # support either name
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    s._timeout = request_timeout  # type: ignore[attr-defined]
    s._max_retries = max_retries  # type: ignore[attr-defined]
    return s

def _get_json(sess: requests.Session, url: str) -> dict:
    data = _retry(sess, "GET", url).json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON at {url}")
    return data

def _retry(sess: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    t = getattr(sess, "_timeout", 30.0)
    rmax = int(getattr(sess, "_max_retries", 5))
    backoff = 1.0
    last_exc: Optional[Exception] = None
    for _ in range(rmax):
        try:
            resp = sess.request(method, url, timeout=t, **kwargs)
            if resp.status_code in (429, 502, 503, 504):
                time.sleep(backoff); backoff = min(backoff * 2, 20.0)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            time.sleep(backoff); backoff = min(backoff * 2, 20.0)
    raise RuntimeError(f"Request failed for {url}: {type(last_exc).__name__}: {last_exc}")

def _iter_pages(sess: requests.Session, url: str) -> Iterable[dict]:
    while url:
        payload = _get_json(sess, url)
        yield payload
        url = (payload.get("links") or {}).get("next")

def _walk_folder(sess: requests.Session, folder_href: str) -> Iterable[dict]:
    # Depth-first traversal of osfstorage
    for page in _iter_pages(sess, folder_href):
        for item in page.get("data", []):
            attrs = item.get("attributes") or {}
            kind = (attrs.get("kind") or "").lower()
            if kind == "folder":
                next_href = ((item.get("links") or {}).get("related") or {}).get("href")
                if not next_href:
                    next_href = (
                        ((item.get("relationships") or {}).get("files") or {})
                        .get("links", {}).get("related", {}).get("href")
                    )
                if next_href:
                    yield from _walk_folder(sess, next_href)
            else:
                yield item

def _head_content_length(sess: requests.Session, url: str) -> Optional[int]:
    try:
        resp = _retry(sess, "HEAD", url, allow_redirects=True)
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None

def _stream_download(
    sess: requests.Session,
    url: str,
    out_path: Path,
    *,
    max_left: int,
    chunk_size: int = 1024 * 1024,
) -> Tuple[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    bytes_written = 0
    with _retry(sess, "GET", url, stream=True, allow_redirects=True) as r:
        with open(tmp_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                if bytes_written + len(chunk) > max_left:
                    fh.flush(); fh.close()
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError("Exceeded max_bytes during file download")
                fh.write(chunk)
                bytes_written += len(chunk)

    shutil.move(str(tmp_path), str(out_path))
    return str(out_path), bytes_written

def _infer_download(node: dict) -> Optional[str]:
    links = node.get("links") or {}
    href = links.get("download") or links.get("href") or links.get("self")
    if href:
        # WaterButler file endpoints accept ?action=download
        sep = "&" if "?" in href else "?"
        return f"{href}{sep}action=download"
    return None

def _sanitize_rel_path(materialized_path: str) -> str:
    p = Path(materialized_path.strip("/"))
    parts = []
    for part in p.parts:
        if part in (".", ".."):
            continue
        parts.append(part.replace("\x00", "").strip())
    if not parts:
        parts = ["unnamed"]
    return str(Path(*parts))
