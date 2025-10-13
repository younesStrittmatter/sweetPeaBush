# sweetExtract/steps/build_unpacked_data.py
from __future__ import annotations
import os, json, hashlib, shutil, zipfile, tarfile, gzip, bz2, lzma
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project

IGNORED = {".DS_Store", ".gitkeep"}
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")
_SINGLE_COMPRESSED = {".gz", ".bz2", ".xz"}  # (but not tarballs)

def _is_hidden_or_ignored(p: Path) -> bool:
    n = p.name
    return n.startswith(".") or n in IGNORED

def _is_archive(p: Path) -> bool:
    return p.name.lower().endswith(_ARCHIVE_SUFFIXES)

def _is_single_compressed(p: Path) -> bool:
    sfx = p.suffix.lower()
    if sfx not in _SINGLE_COMPRESSED:
        return False
    low = str(p).lower()
    return not (low.endswith(".tar.gz") or low.endswith(".tar.bz2") or low.endswith(".tar.xz"))

def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def _safe_join(base: Path, *parts: str) -> Path:
    out = base.joinpath(*parts).resolve()
    if not str(out).startswith(str(base.resolve())):
        raise ValueError(f"Unsafe path traversal prevented: {out}")
    return out

def _decompress_single(src: Path, dst: Path) -> None:
    opener = gzip.open if src.suffix.lower() == ".gz" else bz2.open if src.suffix.lower() == ".bz2" else lzma.open
    dst.parent.mkdir(parents=True, exist_ok=True)
    with opener(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)

def _strip_single_compress_ext(p: Path) -> Path:
    return p.with_suffix("")  # foo.csv.gz -> foo.csv

def _find_alt_path(dst: Path) -> Path:
    stem = dst.stem
    suf = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__alt{i}{suf}"
        if not candidate.exists():
            return candidate
        i += 1

class BuildUnpackedData(BaseStep):
    """
    Consolidate everything into artifacts/data_unpacked/ without extra 'raw' nesting:
      • Plain files -> data_unpacked/<relpath-from-data_raw>
      • .gz/.bz2/.xz -> data_unpacked/<relpath-without-compress-ext>
      • Archives -> data_unpacked/archives/<archive_stem>/<member_path>
    Deduplicate by SHA-256 across all outputs. Avoid clobbering by using __altN when needed.

    Artifact: artifacts/meta/data_unpacked_manifest.json
    """

    def __init__(self, force: bool = False):
        super().__init__(
            name="build_unpacked_data",
            artifact="meta/data_unpacked_manifest.json",
            depends_on=[],
            map_over=None,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        if self._force:
            return True
        # Only run if there is raw data and our manifest doesn't exist yet
        out_path = project.artifacts_dir / self.artifact
        return project.has_data_raw_files() and not out_path.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        art_root = project.artifacts_dir.resolve()
        raw_root = project.data_raw_dir.resolve()
        dest_root = (art_root / "data_unpacked")
        dest_root.mkdir(parents=True, exist_ok=True)

        entries: List[Dict[str, Any]] = []
        seen_hash: Dict[str, str] = {}  # sha256 -> dest_relpath
        bytes_written = n_copied = n_linked = n_decompressed = n_extracted = n_dedup = n_alt = 0

        # --- Pass 1: archives ---
        for arc in sorted([p for p in raw_root.rglob("*") if p.is_file() and _is_archive(p)]):
            arc_rel = str(arc.relative_to(raw_root))
            base_out = _safe_join(dest_root, "archives", arc.stem)
            try:
                if arc.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(arc) as z:
                        for m in z.infolist():
                            if m.is_dir():
                                continue
                            target = _safe_join(base_out, m.filename)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with z.open(m.filename) as src, target.open("wb") as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)
                            sha = _sha256(target)
                            size = target.stat().st_size
                            if sha in seen_hash:
                                try: target.unlink()
                                except Exception: pass
                                entries.append({
                                    "source": "archive", "archive_relpath": arc_rel,
                                    "member": m.filename, "dest": seen_hash[sha],
                                    "sha256": sha, "size": size, "action": "dedup_skipped"
                                })
                                n_dedup += 1
                            else:
                                rel_dst = str(target.relative_to(art_root))
                                seen_hash[sha] = rel_dst
                                bytes_written += size
                                n_extracted += 1
                                entries.append({
                                    "source": "archive", "archive_relpath": arc_rel,
                                    "member": m.filename, "dest": rel_dst,
                                    "sha256": sha, "size": size, "action": "extracted"
                                })
                else:
                    mode = ("r:" if arc.name.endswith(".tar") else
                            "r:gz" if arc.name.endswith((".tgz", ".tar.gz")) else
                            "r:bz2" if arc.name.endswith((".tbz2", ".tar.bz2")) else
                            "r:xz")
                    with tarfile.open(arc, mode) as t:
                        for m in t.getmembers():
                            if not m.isfile():
                                continue
                            target = _safe_join(base_out, m.name)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with t.extractfile(m) as src, target.open("wb") as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)
                            sha = _sha256(target)
                            size = target.stat().st_size
                            if sha in seen_hash:
                                try: target.unlink()
                                except Exception: pass
                                entries.append({
                                    "source": "archive", "archive_relpath": arc_rel,
                                    "member": m.name, "dest": seen_hash[sha],
                                    "sha256": sha, "size": size, "action": "dedup_skipped"
                                })
                                n_dedup += 1
                            else:
                                rel_dst = str(target.relative_to(art_root))
                                seen_hash[sha] = rel_dst
                                bytes_written += size
                                n_extracted += 1
                                entries.append({
                                    "source": "archive", "archive_relpath": arc_rel,
                                    "member": m.name, "dest": rel_dst,
                                    "sha256": sha, "size": size, "action": "extracted"
                                })
            except Exception as e:
                entries.append({
                    "source": "archive", "archive_relpath": arc_rel,
                    "error": f"{type(e).__name__}: {e}", "action": "error"
                })

        # --- Pass 2: single-file compressed (.gz/.bz2/.xz, not tarballs) ---
        for f in sorted([p for p in raw_root.rglob("*") if p.is_file() and _is_single_compressed(p)]):
            src_rel = str(f.relative_to(raw_root))
            out_rel = _strip_single_compress_ext(f.relative_to(raw_root))
            dst = _safe_join(dest_root, str(out_rel))
            tmp = dst.with_suffix(dst.suffix + ".tmp_dec")
            try:
                _decompress_single(f, tmp)
                sha = _sha256(tmp)
                size = tmp.stat().st_size
                if sha in seen_hash:
                    try: tmp.unlink()
                    except Exception: pass
                    entries.append({
                        "source": "single_compressed", "src": src_rel,
                        "dest": seen_hash[sha], "sha256": sha, "size": size,
                        "action": "dedup_skipped"
                    })
                    n_dedup += 1
                    continue

                # avoid clobber: if dst exists with different content, choose alt
                if dst.exists():
                    try:
                        if _sha256(dst) == sha:
                            # identical; skip writing
                            try: tmp.unlink()
                            except Exception: pass
                            entries.append({
                                "source": "single_compressed", "src": src_rel,
                                "dest": str(dst.relative_to(art_root)), "sha256": sha,
                                "size": size, "action": "dedup_skipped"
                            })
                            n_dedup += 1
                            continue
                    except Exception:
                        pass
                    dst = _find_alt_path(dst)
                    n_alt += 1

                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp.rename(dst)
                rel_dst = str(dst.relative_to(art_root))
                seen_hash[sha] = rel_dst
                bytes_written += size
                n_decompressed += 1
                entries.append({
                    "source": "single_compressed", "src": src_rel,
                    "dest": rel_dst, "sha256": sha, "size": size, "action": "decompressed"
                })
            except Exception as e:
                try:
                    if tmp.exists(): tmp.unlink()
                except Exception:
                    pass
                entries.append({
                    "source": "single_compressed", "src": src_rel,
                    "error": f"{type(e).__name__}: {e}", "action": "error"
                })

        # --- Pass 3: plain files (no archive, no single-compressed) ---
        plain_files = [
            p for p in raw_root.rglob("*")
            if p.is_file() and not _is_hidden_or_ignored(p) and not _is_archive(p) and not _is_single_compressed(p)
        ]
        for f in sorted(plain_files):
            src_rel = str(f.relative_to(raw_root))
            dst = _safe_join(dest_root, src_rel)
            sha = _sha256(f)
            size = f.stat().st_size
            if sha in seen_hash:
                entries.append({
                    "source": "plain", "src": src_rel,
                    "dest": seen_hash[sha], "sha256": sha, "size": size,
                    "action": "dedup_skipped"
                })
                n_dedup += 1
                continue

            # avoid clobber: if dst exists with different content, pick alt
            if dst.exists():
                try:
                    if _sha256(dst) == sha:
                        entries.append({
                            "source": "plain", "src": src_rel,
                            "dest": str(dst.relative_to(art_root)), "sha256": sha,
                            "size": size, "action": "dedup_skipped"
                        })
                        n_dedup += 1
                        continue
                except Exception:
                    pass
                dst = _find_alt_path(dst)
                n_alt += 1

            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(f, dst)
                action = "linked"; n_linked += 1
            except Exception:
                shutil.copy2(f, dst)
                action = "copied"; n_copied += 1; bytes_written += size

            rel_dst = str(dst.relative_to(art_root))
            seen_hash[sha] = rel_dst
            entries.append({
                "source": "plain", "src": src_rel,
                "dest": rel_dst, "sha256": sha, "size": size, "action": action
            })

        return {
            "dest_root": str(dest_root),
            "entries": entries,
            "summary": {
                "unique_files": len(seen_hash),
                "extracted": n_extracted,
                "decompressed": n_decompressed,
                "copied": n_copied,
                "linked": n_linked,
                "dedup_skipped": n_dedup,
                "path_alternatives": n_alt,
                "bytes_written": bytes_written,
            },
            "notes": [
                "Plain and decompressed files are placed directly under artifacts/data_unpacked/ preserving raw relpaths (compression suffix removed).",
                "Archive members live under artifacts/data_unpacked/archives/<archive_stem>/ to avoid collisions.",
                "Content-hash dedup across all sources prevents duplicates; __altN is used when different files collide by path.",
            ],
        }
