# sweetExtract/steps/asset_catalog.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.unpack_data import BuildUnpackedData
from sweetExtract.steps.catalog import Catalog

# ----------------------- config -----------------------

MAX_ASSET_FILES_LISTED = 5000    # hard cap on enumerated asset records
MAX_EXAMPLES_PER_KIND  = 40      # short example lists per category
PATHLIKE_SUBSTRS = [
    "img", "image", "file", "path", "stim", "sprite", "mask",
    "movie", "video", "frame", "audio", "sound", "wav", "mp3",
    "ogg", "mp4", "webm", "jpg", "jpeg", "png", "gif", "svg", "bmp", "tif", "tiff"
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".opus"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".ogv"}

# Everything not text/excel/json from Catalog and not in the above sets will fall into "other_binary".


# ----------------------- helpers -----------------------

def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _asset_kind_for_ext(ext: str) -> str:
    e = ext.lower()
    if e in IMAGE_EXTS: return "image"
    if e in AUDIO_EXTS: return "audio"
    if e in VIDEO_EXTS: return "video"
    return "other_binary"

def _collect_from_fs(base: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    """
    Return (files_list, by_ext_count, total_bytes) scanning data_unpacked.
    Each file record: {relpath, size_bytes, ext, kind}
    """
    files: List[Dict[str, Any]] = []
    by_ext: Dict[str, int] = {}
    total_bytes = 0

    if not base.exists():
        return files, by_ext, total_bytes

    listed = 0
    for p in sorted(base.rglob("*")):
        if not p.is_file():      # skip dirs
            continue
        if p.name.startswith("."):  # skip hidden
            continue

        ext = p.suffix.lower()
        kind = _asset_kind_for_ext(ext)
        size = int(p.stat().st_size)
        by_ext[ext] = by_ext.get(ext, 0) + 1
        total_bytes += size

        # Only list detailed rows up to a cap (summary still includes all via by_ext/total_bytes)
        if listed < MAX_ASSET_FILES_LISTED:
            files.append({
                "relpath": str(p.relative_to(base)),
                "size_bytes": size,
                "ext": ext,
                "kind": kind
            })
            listed += 1

    return files, by_ext, total_bytes

def _candidate_columns_from_catalog(catalog_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Mine meta/catalog.json to find columns/keys that *look* like they may hold paths to assets.
    Returns a list of {file, columns} (or {file, sheet, columns} for Excel).
    """
    out: List[Dict[str, Any]] = []
    files = catalog_obj.get("files") or []
    for rec in files:
        kind = rec.get("kind")
        ext  = (rec.get("ext") or "").lower()
        rel  = rec.get("relpath") or ""
        # text tables
        if kind == "table/text":
            cols = [c for c in (rec.get("columns_head") or []) if _has_pathlike_name(c)]
            if cols:
                out.append({"file": rel, "columns": cols})
        # excel tables (per sheet headers)
        elif kind == "table/excel":
            sheets = rec.get("sheet_headers") or {}
            for sheet_name, cols in (sheets.items() if isinstance(sheets, dict) else []):
                hit = [c for c in (cols or []) if _has_pathlike_name(c)]
                if hit:
                    out.append({"file": rel, "sheet": sheet_name, "columns": hit})
        # json top-level keys sample
        elif kind == "json":
            keys = rec.get("top_level_keys_sample") or []
            hit = [k for k in keys if _has_pathlike_name(k)]
            if hit:
                out.append({"file": rel, "keys": hit})
        else:
            # ignore "other" kinds here; actual binaries are discovered via FS scan
            pass
    return out

def _has_pathlike_name(name: str) -> bool:
    s = (name or "").strip().lower()
    return any(tok in s for tok in PATHLIKE_SUBSTRS)

def _examples(files: List[Dict[str, Any]], target_kind: str, limit: int = MAX_EXAMPLES_PER_KIND) -> List[str]:
    return [f["relpath"] for f in files if f.get("kind") == target_kind][:limit]


# ----------------------- the step -----------------------

class AssetCatalog(BaseStep):
    """
    Build a compact catalog of *binary* assets under artifacts/data_unpacked/:
      - Images, Audio, Video, and Other binaries
      - Summaries by ext/kind, total bytes
      - Example file names
      - Candidate table/JSON columns that look like they contain file paths

    Artifact: artifacts/meta/asset_catalog.json
    """

    def __init__(self, force: bool = False):
        super().__init__(
            name="asset_catalog",
            artifact="meta/asset_catalog.json",
            depends_on=[BuildUnpackedData, Catalog],  # ensure unpack + table catalog first
            map_over=None,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        if self._force:
            return True
        out = (project.artifacts_dir / self.artifact)
        return not out.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        art_root = project.artifacts_dir.resolve()
        base = (art_root / "data_unpacked").resolve()

        # 1) Scan filesystem for binaries
        files, by_ext, total_bytes = _collect_from_fs(base)

        # 2) Load meta/catalog.json to mine path-like columns/keys
        cat = _read_json(art_root / "meta" / "catalog.json") or {}
        path_columns = _candidate_columns_from_catalog(cat)

        # 3) Aggregate counts by kind
        by_kind: Dict[str, int] = {"image": 0, "audio": 0, "video": 0, "other_binary": 0}
        for rec in files:
            k = rec.get("kind")
            if k in by_kind:
                by_kind[k] += 1

        # 4) Build per-kind summaries with short example lists
        images_examples = _examples(files, "image")
        audio_examples  = _examples(files, "audio")
        video_examples  = _examples(files, "video")
        other_examples  = _examples(files, "other_binary")

        result: Dict[str, Any] = {
            "root": str(base),
            "summary": {
                "counts": by_kind,
                "by_ext": [{"ext": k, "count": v} for k, v in sorted(by_ext.items())],
                "total_bytes": total_bytes,
            },
            "assets": {
                "images": {
                    "count": by_kind["image"],
                    "examples": images_examples,
                },
                "audio": {
                    "count": by_kind["audio"],
                    "examples": audio_examples,
                },
                "video": {
                    "count": by_kind["video"],
                    "examples": video_examples,
                },
                "other_binary": {
                    "count": by_kind["other_binary"],
                    "examples": other_examples,
                },
            },
            "files": files,  # truncated to MAX_ASSET_FILES_LISTED
            "link_hints": {
                "possible_path_columns": path_columns,
                "patterns_used": PATHLIKE_SUBSTRS,
                "notes": [
                    "Columns/keys are detected by substring match only; values were not parsed.",
                    "Prefer columns as asset sources when available (more robust than raw filenames)."
                ]
            },
        }

        out = (art_root / "meta" / "asset_catalog.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
