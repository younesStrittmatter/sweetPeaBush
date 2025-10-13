# sweetExtract/steps/probe_candidate_tables.py
from __future__ import annotations
import json, os, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments
from sweetExtract.steps.llm_propose_trial_candidates import LLMProposeTrialCandidates

# Small, conservative limits
_DEFAULT_SAMPLE_ROWS = int(os.getenv("SWEETEXTRACT_PROBE_NROWS", "120"))
_DEFAULT_MAX_BYTES   = int(os.getenv("SWEETEXTRACT_PROBE_MAX_BYTES", "20000000"))  # 20MB per file cap

_SUBJECT_COL_CANDIDATES = [
    "subject","participant","participant_id","subj","sid","id","S","Subject","Participant"
]
_EXPERIMENT_COL_CANDIDATES = [
    "experiment","exp","dataset","study","task","condition","group","cohort","Experiment","Exp"
]
_TRIAL_COL_CANDIDATES = [
    "trial","trial_index","trialnum","trialn","ntrial","Trial","TrialN","TrialNum"
]

def _read_table_sample(root: Path, relpath: str, sheet: str, fmt: str, nrows: int
                      ) -> Tuple[List[str], List[List[Any]], Dict[str, int]]:
    """
    Returns (columns, rows_sampled, file_stats) or ([], [], stats) on failure.
      - columns: list[str]
      - rows_sampled: list[list[Any]] (up to 3 preview rows; values stringified)
      - file_stats: {'n_rows': int or -1, 'n_cols': int or 0, 'error': str or ''}

    Reads head only; respects a very small byte limit to avoid heavy files.
    """
    import pandas as pd

    p = (root / relpath).resolve()
    stats = {"n_rows": -1, "n_cols": 0, "error": ""}

    if not p.exists():
        stats["error"] = "missing_file"
        return [], [], stats
    try:
        if p.stat().st_size > _DEFAULT_MAX_BYTES:
            stats["error"] = f"too_large:{p.stat().st_size}"
            return [], [], stats
    except Exception as e:
        stats["error"] = f"stat_error:{type(e).__name__}"
        return [], [], stats

    try:
        if fmt in ("csv","tsv"):
            sep = "\t" if fmt == "tsv" else None
            df = pd.read_csv(p, nrows=nrows, sep=sep, engine="python")
        elif fmt == "excel":
            df = pd.read_excel(p, sheet_name=(sheet or 0), nrows=nrows, engine=None)
        elif fmt == "json":
            # try to load line-delimited or array; fallback to pandas read_json with chunks
            try:
                df = pd.read_json(p, lines=True)  # often ndjson
            except Exception:
                df = pd.read_json(p)              # plain JSON array/object
            if nrows and len(df) > nrows:
                df = df.head(nrows)
        else:
            stats["error"] = f"unsupported_format:{fmt}"
            return [], [], stats

        cols = [str(c) for c in df.columns]
        stats["n_rows"] = int(len(df))
        stats["n_cols"] = int(len(cols))

        # tiny preview (up to 3 rows)
        preview_rows: List[List[Any]] = []
        for _, row in df.head(3).iterrows():
            preview_rows.append([None if pd.isna(v) else (str(v) if not isinstance(v, (int,float)) else v) for v in row.tolist()])

        return cols, preview_rows, stats
    except Exception as e:
        stats["error"] = f"read_error:{type(e).__name__}:{e}"
        return [], [], stats


def _present_cols(cols: List[str], wanted: List[str]) -> List[str]:
    want_l = {w.lower() for w in wanted}
    return [c for c in cols if c.lower() in want_l]

# ---- alias helpers -----------------------------------------------------------

def _append_alias(project: Project, canonical_title: str, alias_title: str, source_step: str) -> None:
    """
    Append an alias for an experiment into artifacts/meta/experiment_aliases.json.
    Idempotent (no duplicates).
    """
    alias_title = (alias_title or "").strip()
    canonical_title = (canonical_title or "").strip()
    if not alias_title or alias_title == canonical_title:
        return

    path = project.artifacts_dir / "meta" / "experiment_aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"items": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {"items": []}
        except Exception:
            data = {"items": []}

    items = data.get("items")
    if not isinstance(items, list):
        items = []
        data["items"] = items

    rec = None
    for it in items:
        if isinstance(it, dict) and it.get("experiment_title") == canonical_title:
            rec = it
            break
    if rec is None:
        rec = {"experiment_title": canonical_title, "aliases": []}
        items.append(rec)

    aliases = rec.get("aliases")
    if not isinstance(aliases, list):
        aliases = []
        rec["aliases"] = aliases

    if alias_title not in aliases:
        aliases.append(alias_title)

    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    updates = meta.get("updates")
    if not isinstance(updates, list):
        updates = []
        meta["updates"] = updates
    updates.append({
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "step": source_step,
        "experiment_title": canonical_title,
        "added_alias": alias_title,
    })

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class ProbeCandidateTables(BaseStep):
    """
    For each experiment, open a small sample from the LLM-proposed top candidates and
    record verified facts: columns present, small preview, and light stats.

    Input:
      - artifacts/meta/llm_propose_trial_candidates.json (per-experiment candidates)
      - artifacts/data_unpacked/ (resolved via relpath)
      - DescribeExperiments (for canonical titles; aliases recorded automatically)

    Output:
      - artifacts/meta/candidate_probes.json (combined)
      - artifacts/probe_candidate_tables/{idx}.json (per experiment)
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, top_k: int = 3, force: bool = False):
        super().__init__(
            name="probe_candidate_tables",
            artifact="meta/candidate_probes.json",
            depends_on=[DescribeExperiments, LLMProposeTrialCandidates],  # ensure proposals exist in order
            map_over=DescribeExperiments,
        )
        self.top_k = top_k
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        proposals = project.artifacts_dir / "meta" / "llm_propose_trial_candidates.json"
        out = project.artifacts_dir / self.artifact
        return proposals.exists() and (self._force or not out.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError  # mapped; compute_one handles per experiment

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        art = project.artifacts_dir
        unpacked = project.artifacts_dir / "data_unpacked"

        # Align with proposals by index (same map_over base)
        proposals_all = json.loads((art / "meta" / "llm_propose_trial_candidates.json").read_text(encoding="utf-8"))
        cand_block = None
        if isinstance(proposals_all, dict) and isinstance(proposals_all.get("items"), list) and idx < len(proposals_all["items"]):
            cand_block = proposals_all["items"][idx]
        elif isinstance(proposals_all, list) and idx < len(proposals_all):
            cand_block = proposals_all[idx]

        canonical_title = item.get("title") or f"Experiment {idx+1}"
        out: Dict[str, Any] = {
            "experiment_title": canonical_title,
            "probes": [],
            "notes": [],
        }

        # If proposals used a different title, record it as an alias (no extra step)
        alt_title = (cand_block or {}).get("experiment_title")
        if alt_title and alt_title != canonical_title:
            _append_alias(project, canonical_title=canonical_title, alias_title=alt_title, source_step=self.name)
            out["notes"].append(f"added_alias:{alt_title}")

        if not cand_block:
            out["notes"].append("no_candidates_for_experiment")
            return out

        cands = cand_block.get("candidates") or []
        for c in cands[: self.top_k]:
            relpath = c.get("relpath") or ""
            fmt = (c.get("format") or "other").lower()
            sheet = c.get("sheet") or ""
            # normalize format from extension if missing
            if fmt == "other":
                ext = Path(relpath).suffix.lower()
                if ext == ".csv": fmt = "csv"
                elif ext in (".tsv",".tab"): fmt = "tsv"
                elif ext in (".xlsx",".xls"): fmt = "excel"
                elif ext == ".json": fmt = "json"

            cols, preview, stats = _read_table_sample(unpacked, relpath, sheet, fmt, _DEFAULT_SAMPLE_ROWS)

            present_subject = _present_cols(cols, _SUBJECT_COL_CANDIDATES)
            present_trial   = _present_cols(cols, _TRIAL_COL_CANDIDATES)
            present_expm    = _present_cols(cols, _EXPERIMENT_COL_CANDIDATES)

            out["probes"].append({
                "relpath": relpath,
                "sheet": sheet,
                "format": fmt,
                "columns": cols,
                "present_subject_cols": present_subject,
                "present_trial_cols": present_trial,
                "present_experiment_cols": present_expm,
                "preview_rows": preview,
                "stats": stats,
            })

        return out
