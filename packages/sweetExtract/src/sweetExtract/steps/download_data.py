# sweetExtract/steps/download_data_if_missing.py
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.find_data_links import FindDataLinks  # depend on the step, not a path
from sweetExtract.utils.download_data.osf_download import download_osf_project

_OSF_ID_RX = re.compile(r"/([a-z0-9]{5})(?:/|$)", re.I)


def _load_sources(artifacts_dir: Path) -> List[Dict[str, Any]]:
    src_file = artifacts_dir / "meta" / "data_sources.json"
    if not src_file.exists():
        return []
    try:
        payload = json.loads(src_file.read_text(encoding="utf-8"))
        return payload.get("sources", []) or []
    except Exception:
        return []


def _pick_osf_source(sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    osf = [s for s in sources if (s.get("provider") or "").lower() == "osf"]
    if not osf:
        return None
    osf.sort(
        key=lambda s: (
            float(s.get("confidence") or 0.0),
            1 if (s.get("canonical_id") or _OSF_ID_RX.search(s.get("url") or "")) else 0
        ),
        reverse=True
    )
    return osf[0]


def _resolve_osf_id(src: Dict[str, Any]) -> Optional[str]:
    cand = (src.get("canonical_id") or "").strip()
    if cand and re.fullmatch(r"[a-z0-9]{5}", cand, flags=re.I):
        return cand.lower()
    m = _OSF_ID_RX.search(cand)
    if m:
        return m.group(1).lower()
    m = _OSF_ID_RX.search(src.get("url") or "")
    if m:
        return m.group(1).lower()
    return None


class DownloadData(BaseStep):
    """
    Downloads data into artifacts/data_raw.

    Default behavior: run only if data_raw is currently empty/missing and sources exist.
    When constructed with force=True, always run (even if data_raw is non-empty).
    Reads sources from artifacts/meta/data_sources.json (produced by FindDataLinks).

    Artifact: artifacts/meta/download_status.json
    """

    def __init__(self, force: bool = False):
        super().__init__(
            name="download_data_if_missing",
            artifact="meta/download_status.json",
            depends_on=[FindDataLinks],  # ensure source links exist (or were attempted)
            map_over=None,
        )
        self._force = force

    # Run only when data_raw is empty/missing, unless force=True
    def should_run(self, project: Project) -> bool:
        if self._force:
            return True
        art = project.artifacts_dir
        sources = _load_sources(art)
        return not project.has_data_raw_files() and bool(sources)

    def compute(self, project: Project) -> Dict[str, Any]:
        data_raw = project.data_raw_dir
        art = project.artifacts_dir
        max_bytes = int(os.getenv("SWEETEXTRACT_MAX_DOWNLOAD_BYTES", "1000000000") or "1000000000")
        sources = _load_sources(art)

        if not sources:
            return {
                "status": "skipped",
                "reason": "no sources found",
                "forced": bool(self._force),
                "max_bytes": max_bytes,
            }

        src = _pick_osf_source(sources)
        if not src:
            return {
                "status": "skipped",
                "reason": "no supported providers (need osf)",
                "sources_seen": sources,
                "forced": bool(self._force),
            }

        osf_id = _resolve_osf_id(src)
        if not osf_id:
            return {"status": "error", "reason": "could not resolve OSF id", "source": src, "forced": bool(self._force)}

        # Optionally warn if forced and folder already has files
        warn = None
        if self._force and project.has_data_raw_files():
            warn = "data_raw not empty; proceeding due to force=True (downloader behavior determines overwrite/skip)."

        try:
            result = download_osf_project(
                osf_id=osf_id,
                dest_dir=str(data_raw),
                max_bytes=max_bytes,
            )
            payload = {
                "status": "ok",
                "provider": "osf",
                "osf_id": osf_id,
                "max_bytes": max_bytes,
                "downloaded_bytes": (result or {}).get("downloaded_bytes"),
                "files": (result or {}).get("files"),
                "forced": bool(self._force),
            }
            if warn:
                payload["note"] = warn
            return payload
        except Exception as e:
            return {
                "status": "error",
                "reason": f"{type(e).__name__}: {e}",
                "provider": "osf",
                "osf_id": osf_id,
                "forced": bool(self._force),
            }
