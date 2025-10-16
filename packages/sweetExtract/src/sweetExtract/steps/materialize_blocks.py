# sweetExtract/steps/materialize_blocks.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import pandas as pd

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.execute_trial_split import ExecuteTrialSplit
from sweetExtract.steps.consolidate_blocks import ConsolidateBlocks

# ---------- small utils ----------

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s or "untitled"

def _equal_chunk_sizes(n: int, k: int) -> List[int]:
    """Return k sizes that sum to n and differ by at most 1 (remainder to the front)."""
    base, r = divmod(n, k)
    return [base + (1 if i < r else 0) for i in range(k)]

# ---------- artifacts readers ----------

def _load_consolidated_map(project: Project) -> Dict[str, Dict[str, Any]]:
    p = project.artifacts_dir / "meta" / "blocks_consolidated.json"
    obj = _read_json(p) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for it in (obj.get("items") or []):
        title = (it or {}).get("experiment_title") or (it or {}).get("title")
        if isinstance(title, str) and title:
            out[title] = it
    return out

def _load_trials_index(project: Project) -> List[Dict[str, Any]]:
    p = project.artifacts_dir / "meta" / "trials_index.json"
    obj = _read_json(p) or {}
    return obj.get("items") or []

def _find_trials_entry(trials_items: List[Dict[str, Any]], title: str) -> Optional[Dict[str, Any]]:
    title_norm = (title or "").strip().lower()
    # exact title
    for it in trials_items:
        t = (it or {}).get("experiment_title") or ""
        if isinstance(t, str) and t.strip().lower() == title_norm:
            return it
    # alias
    for it in trials_items:
        aliases = (it or {}).get("aliases") or []
        for a in aliases if isinstance(aliases, list) else []:
            if isinstance(a, str) and a.strip().lower() == title_norm:
                return it
    # slug fallback
    slug_norm = _slug(title).lower()
    for it in trials_items:
        if _slug((it or {}).get("experiment_title") or "").lower() == slug_norm:
            return it
    return None

# ---------- block partitioning ----------

def _iter_blocks_by_split_column(df: pd.DataFrame, col: str) -> Iterable[Tuple[int, str, pd.DataFrame]]:
    """
    Yield (block_index, block_value, df_block) in the order of first appearance of each value.
    Values are treated as opaque strings; preserves dataset order within each block.
    """
    vals = pd.unique(df[col].astype("string"))
    for i, v in enumerate(vals):
        yield i, str(v), df[df[col].astype("string") == v].copy()

def _iter_blocks_by_index(df: pd.DataFrame, n_blocks: int, block_size: Optional[int]) -> Iterable[Tuple[int, Optional[str], pd.DataFrame]]:
    """
    Yield (block_index, block_value=None, df_block) as contiguous index-based chunks.
    If block_size is None, split equally by index into n_blocks.
    """
    n = len(df)
    if n_blocks <= 0 or n <= 0:
        yield 0, None, df.copy()
        return

    if isinstance(block_size, int) and block_size > 0:
        edges = [min((i + 1) * block_size, n) for i in range(n_blocks)]
        sizes = [edges[0]] + [edges[i] - edges[i - 1] for i in range(1, len(edges))]
        total = sum(sizes)
        if total < n:
            sizes[-1] += (n - total)
    else:
        sizes = _equal_chunk_sizes(n, n_blocks)

    start = 0
    for i, sz in enumerate(sizes):
        end = min(start + sz, n)
        if end <= start:
            continue
        yield i, None, df.iloc[start:end].copy()
        start = end

def _partition_subject_df(df: pd.DataFrame, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return a list of blocks, each as {index, value, df}. Priority:
      1) split_by.column
      2) split_by_index
      3) single block (whole df)
    """
    blocks: List[Dict[str, Any]] = []

    split_by = (plan or {}).get("split_by") or None
    if isinstance(split_by, dict) and isinstance(split_by.get("column"), str):
        col = split_by["column"]
        if col in df.columns:
            for idx, val, part in _iter_blocks_by_split_column(df, col):
                blocks.append({"index": idx, "value": val, "df": part})
            if blocks:
                return blocks  # done

    sbi = (plan or {}).get("split_by_index") or None
    if isinstance(sbi, dict) and isinstance(sbi.get("n_blocks"), int) and sbi["n_blocks"] > 0:
        n_blocks = int(sbi["n_blocks"])
        block_size = sbi.get("block_size")
        for idx, val, part in _iter_blocks_by_index(df, n_blocks, block_size if isinstance(block_size, int) else None):
            blocks.append({"index": idx, "value": val, "df": part})
        if blocks:
            return blocks

    # single block fallback
    blocks.append({"index": 0, "value": None, "df": df.copy()})
    return blocks

# ---------- the step ----------

class MaterializeBlocks(BaseStep):
    """
    Use the previous per-subject trial CSVs (ExecuteTrialSplit → trials_index.json),
    apply the consolidated plan (ConsolidateBlocks), and write one CSV **per block**:

      artifacts/trials_blocks/<experiment_slug>/subject_<id>/block0.csv
      artifacts/trials_blocks/<experiment_slug>/subject_<id>/block1.csv
      ...

    Also writes a QC summary:
      artifacts/meta/blocks_materialized.json
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="materialize_blocks",
            artifact="meta/blocks_materialized.json",
            depends_on=[ExecuteTrialSplit, ConsolidateBlocks],
            map_over=ConsolidateBlocks,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        dst = project.artifacts_dir / self.artifact
        trials_idx = project.artifacts_dir / "meta" / "trials_index.json"
        blocks_con = project.artifacts_dir / "meta" / "blocks_consolidated.json"
        return trials_idx.exists() and blocks_con.exists() and (self._force or not dst.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(
        self,
        project: Project,
        item: Dict[str, Any],
        idx: int,
        all_items: List[Dict[str, Any]],
        prior_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        title = item.get("experiment_title") or item.get("title") or f"Experiment {idx+1}"

        # Load artifacts
        con_map = _load_consolidated_map(project)
        con_item = con_map.get(title) or {}
        plan = (con_item.get("plan") or {})
        strategy = con_item.get("strategy") or "unknown"

        trials_items = _load_trials_index(project)
        tr_entry = _find_trials_entry(trials_items, title)
        if not tr_entry:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_trials_entry_for_title",
                "strategy": strategy,
                "plan": plan,
            }

        slug = tr_entry.get("slug") or _slug(title)
        subjects = tr_entry.get("subjects") or []
        if not isinstance(subjects, list) or not subjects:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_subject_files",
                "strategy": strategy,
                "plan": plan,
            }

        base_out_dir = project.artifacts_dir / "trials_blocks" / slug
        base_out_dir.mkdir(parents=True, exist_ok=True)

        subjects_preview: Dict[str, Dict[str, Any]] = {}
        saved_paths: List[str] = []

        for s in subjects:
            rel = (s or {}).get("relpath")
            sid = (s or {}).get("subject_id") or "dataset"
            if not isinstance(rel, str) or not rel:
                continue
            src = (project.artifacts_dir / rel).resolve()
            if not src.exists():
                continue

            # Load the subject CSV from ExecuteTrialSplit
            try:
                df = pd.read_csv(src, engine="python")
            except Exception as e:
                subjects_preview[str(sid)] = {"error": f"read_error:{type(e).__name__}:{e}"}
                continue

            # Partition into blocks
            parts = _partition_subject_df(df, plan)
            subj_dir = base_out_dir / f"subject_{_slug(str(sid))}"
            subj_dir.mkdir(parents=True, exist_ok=True)

            block_meta: List[Dict[str, Any]] = []
            for blk in parts:
                bi = int(blk["index"])
                bv = blk["value"] if blk["value"] is None else str(blk["value"])
                bdf = blk["df"]
                dst = subj_dir / f"block{bi}.csv"
                bdf.to_csv(dst, index=False, encoding="utf-8")
                rel_dst = str(dst.relative_to(project.artifacts_dir))
                saved_paths.append(rel_dst)
                block_meta.append({
                    "index": bi,
                    "value": bv,
                    "n_trials": int(len(bdf)),
                    "relpath": rel_dst,
                })

            # QC per subject
            subjects_preview[str(sid)] = {
                "n_trials_total": int(len(df)),
                "n_blocks": len(block_meta),
                "blocks": block_meta,
                "subject_dir": str(subj_dir.relative_to(project.artifacts_dir)),
                "source_relpath": rel,
            }

        exp_preview = {
            "strategy": strategy,
            "split_by": plan.get("split_by"),
            "split_by_index": plan.get("split_by_index"),
            "notes": plan.get("notes"),
            "n_subjects": len(subjects_preview),
            "subjects": subjects_preview,
            "saved_paths": saved_paths or None,
            "dest_root": str(base_out_dir.relative_to(project.artifacts_dir)),
            "source_dir": tr_entry.get("dest_dir"),
        }

        return {
            "experiment_title": title,
            "status": "ok",
            "preview": exp_preview,
        }
