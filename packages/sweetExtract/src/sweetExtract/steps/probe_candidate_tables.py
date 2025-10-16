from __future__ import annotations
import json, os, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_propose_trial_candidates import LLMProposeTrialCandidates

_DEFAULT_SAMPLE_ROWS = int(os.getenv("SWEETEXTRACT_PROBE_NROWS", "120"))
_DEFAULT_MAX_BYTES   = int(os.getenv("SWEETEXTRACT_PROBE_MAX_BYTES", "20000000"))
_TOPK_UNIQUES        = int(os.getenv("SWEETEXTRACT_PROBE_TOPK_UNIQUES", "6"))  # examples from UNIQUE values

_SUBJECT_COL_CANDIDATES = [
    "subject","participant","participant_id","subj","sid","id","S","Subject","Participant"
]
_EXPERIMENT_COL_CANDIDATES = [
    "experiment","exp","dataset","study","task","condition","group","cohort","Experiment","Exp"
]
_TRIAL_COL_CANDIDATES = [
    "trial","trial_index","trialnum","trialn","ntrial","Trial","TrialN","TrialNum"
]

# -------------------- file probing --------------------

def _read_table_sample(root: Path, relpath: str, sheet: str, fmt: str, nrows: int
                      ) -> Tuple[List[str], List[List[Any]], Dict[str, int], "pd.DataFrame|None"]:
    import pandas as pd
    p = (root / relpath).resolve()
    stats = {"n_rows": -1, "n_cols": 0, "error": ""}
    if not p.exists():
        stats["error"] = "missing_file"; return [], [], stats, None
    try:
        if p.stat().st_size > _DEFAULT_MAX_BYTES:
            stats["error"] = f"too_large:{p.stat().st_size}"; return [], [], stats, None
    except Exception as e:
        stats["error"] = f"stat_error:{type(e).__name__}"; return [], [], stats, None
    try:
        if fmt in ("csv","tsv"):
            sep = "\t" if fmt == "tsv" else None
            df = pd.read_csv(p, nrows=nrows, sep=sep, engine="python")
        elif fmt == "excel":
            df = pd.read_excel(p, sheet_name=(sheet or 0), nrows=nrows, engine=None)
        elif fmt == "json":
            try: df = pd.read_json(p, lines=True)
            except Exception: df = pd.read_json(p)
            if nrows and len(df) > nrows: df = df.head(nrows)
        else:
            stats["error"] = f"unsupported_format:{fmt}"; return [], [], stats, None

        cols = [str(c) for c in df.columns]
        stats["n_rows"] = int(len(df)); stats["n_cols"] = int(len(cols))

        preview_rows: List[List[Any]] = []
        for _, row in df.head(3).iterrows():
            preview_rows.append([None if pd.isna(v) else (str(v) if not isinstance(v,(int,float)) else v) for v in row.tolist()])
        return cols, preview_rows, stats, df
    except Exception as e:
        stats["error"] = f"read_error:{type(e).__name__}:{e}"; return [], [], stats, None

def _present_cols(cols: List[str], wanted: List[str]) -> List[str]:
    want_l = {w.lower() for w in wanted}
    return [c for c in cols if c.lower() in want_l]

def _uniq_stats_full_or_sample(root: Path, relpath: str, sheet: str, fmt: str,
                               col: str, sample_df) -> Dict[str, Any]:
    """
    Compute uniqueness over the FULL column when feasible (<= _DEFAULT_MAX_BYTES),
    otherwise over the provided sample_df. Return:
      { unique_count:int, examples:list[str], truncated:bool, source:"full"|"sample" }
    """
    import pandas as pd
    out = {"unique_count": 0, "examples": [], "truncated": False, "source": "sample"}
    p = (root / relpath).resolve()

    def from_series(s: "pd.Series", source: str) -> Dict[str, Any]:
        s = s.dropna().astype(str)
        uniq = s.unique().tolist()
        out_local = {
            "unique_count": int(len(uniq)),
            "examples": [str(x) for x in uniq[: max(_TOPK_UNIQUES, 0)]],
            "truncated": len(uniq) > max(_TOPK_UNIQUES, 0),
            "source": source,
        }
        return out_local

    # Try full scan (lightweight: read only the column)
    try:
        size_ok = p.exists() and p.stat().st_size <= _DEFAULT_MAX_BYTES
    except Exception:
        size_ok = False

    if size_ok:
        try:
            if fmt in ("csv","tsv"):
                sep = "\t" if fmt == "tsv" else None
                s = pd.read_csv(p, usecols=[col], dtype=str, sep=sep, engine="python")[col]
                return from_series(s, "full")
            elif fmt == "excel":
                s = pd.read_excel(p, sheet_name=(sheet or 0), usecols=[col], dtype=str, engine=None)[col]
                return from_series(s, "full")
            elif fmt == "json":
                try:
                    s = pd.read_json(p, lines=True, dtype=str)[col]
                except Exception:
                    s = pd.read_json(p, dtype=str)[col]
                return from_series(s, "full")
        except Exception:
            pass  # fall back to sample

    # Sample-based fallback
    try:
        if sample_df is not None and col in sample_df.columns:
            return from_series(sample_df[col], "sample")
    except Exception:
        pass

    return out

# -------------------- alias helper --------------------

def _append_alias(project: Project, canonical_title: str, alias_title: str, source_step: str) -> None:
    alias_title = (alias_title or "").strip(); canonical_title = (canonical_title or "").strip()
    if not alias_title or alias_title == canonical_title: return
    path = project.artifacts_dir / "meta" / "experiment_aliases.json"; path.parent.mkdir(parents=True, exist_ok=True)
    try: data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) or {"items": []}
    except Exception: data = {"items": []}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list): items = []; data["items"] = items
    rec = next((it for it in items if isinstance(it, dict) and it.get("experiment_title")==canonical_title), None)
    if rec is None: rec = {"experiment_title": canonical_title, "aliases": []}; items.append(rec)
    aliases = rec.get("aliases");
    if not isinstance(aliases, list): aliases = []; rec["aliases"] = aliases
    if alias_title not in aliases: aliases.append(alias_title)
    meta = data.get("meta");
    if not isinstance(meta, dict): meta = {}; data["meta"] = meta
    updates = meta.get("updates");
    if not isinstance(updates, list): updates = []; meta["updates"] = updates
    updates.append({"ts": datetime.datetime.utcnow().isoformat()+"Z","step": source_step,
                    "experiment_title": canonical_title,"added_alias": alias_title})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# -------------------- step --------------------

class ProbeCandidateTables(BaseStep):
    """
    Sample top candidates and record: columns, tiny preview, light stats, and
    uniqueness sketches for subject/experiment ID columns based on FULL columns when feasible.
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, top_k: int = 3, force: bool = False):
        super().__init__(
            name="probe_candidate_tables",
            artifact="meta/candidate_probes.json",
            depends_on=[FilterEmpiricalExperiments, LLMProposeTrialCandidates],
            map_over=FilterEmpiricalExperiments,
        )
        self.top_k = top_k
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        proposals = project.artifacts_dir / "meta" / "llm_propose_trial_candidates.json"
        out = project.artifacts_dir / self.artifact
        return proposals.exists() and (self._force or not out.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        art = project.artifacts_dir
        unpacked = art / "data_unpacked"

        out: Dict[str, Any] = {
            "experiment_title": item.get("title") or f"Experiment {idx+1}",
            "probes": [],
            "notes": [],
        }

        # Load proposals (defensive)
        try:
            proposals_all = json.loads((art / "meta" / "llm_propose_trial_candidates.json").read_text(encoding="utf-8"))
        except Exception as e:
            out["notes"].append(f"proposals_read_error:{type(e).__name__}:{e}")
            return out

        cand_block = None
        if isinstance(proposals_all, dict) and isinstance(proposals_all.get("items"), list) and idx < len(proposals_all["items"]):
            cand_block = proposals_all["items"][idx]
        elif isinstance(proposals_all, list) and idx < len(proposals_all):
            cand_block = proposals_all[idx]

        # Alias tracking
        alt_title = (cand_block or {}).get("experiment_title")
        if alt_title and alt_title != out["experiment_title"]:
            _append_alias(project, out["experiment_title"], alt_title, self.name)
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

            try:
                cols, preview, stats, df = _read_table_sample(unpacked, relpath, sheet, fmt, _DEFAULT_SAMPLE_ROWS)

                present_subject = _present_cols(cols, _SUBJECT_COL_CANDIDATES)
                present_trial   = _present_cols(cols, _TRIAL_COL_CANDIDATES)
                present_expm    = _present_cols(cols, _EXPERIMENT_COL_CANDIDATES)

                subj_uniques: Dict[str, Any] = {}; expm_uniques: Dict[str, Any] = {}
                if cols:
                    for col in present_subject:
                        subj_uniques[col] = _uniq_stats_full_or_sample(unpacked, relpath, sheet, fmt, col, df)
                    for col in present_expm:
                        expm_uniques[col] = _uniq_stats_full_or_sample(unpacked, relpath, sheet, fmt, col, df)

                out["probes"].append({
                    "relpath": relpath, "sheet": sheet, "format": fmt, "columns": cols,
                    "present_subject_cols": present_subject,
                    "present_trial_cols": present_trial,
                    "present_experiment_cols": present_expm,
                    "subject_uniques": subj_uniques, "experiment_uniques": expm_uniques,
                    "preview_rows": preview, "stats": stats,
                })
            except Exception as e:
                out["probes"].append({
                    "relpath": relpath, "sheet": sheet, "format": fmt, "columns": [],
                    "present_subject_cols": [], "present_trial_cols": [], "present_experiment_cols": [],
                    "subject_uniques": {}, "experiment_uniques": {},
                    "preview_rows": [], "stats": {"n_rows": -1, "n_cols": 0, "error": f"probe_error:{type(e).__name__}:{e}"},
                })

        return out
