# sweetExtract/steps/execute_trial_split.py
from __future__ import annotations
import json, re, unicodedata, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments

# ---------- alias helpers (local list kept in trials_index) ----------

_ROMAN = {'M':1000,'CM':900,'D':500,'CD':400,'C':100,'XC':90,'L':50,'XL':40,'X':10,'IX':9,'V':5,'IV':4,'I':1}
_STOP  = {"experiment","experiments","exp","study","studies"}

def _roman_to_int(s: str) -> int | None:
    s = s.upper()
    i = 0; n = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in _ROMAN:
            n += _ROMAN[s[i:i+2]]; i += 2
        elif s[i] in _ROMAN:
            n += _ROMAN[s[i]]; i += 1
        else:
            return None
    return n

def _extract_exp_num(title: str) -> int | None:
    m = re.search(r"\b(?:experiment|exp)\s*([0-9ivxlcdm]+)\b", title or "", flags=re.I)
    if not m: return None
    g = m.group(1)
    return int(g) if g.isdigit() else _roman_to_int(g)

def _basic_norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("—","-").replace("–","-").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks)

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s or "exp"

def _alias_variants(title: str) -> List[str]:
    """Local, lightweight variants for discoverability inside trials_index."""
    title = title or ""
    base  = title.strip()
    no_sub = re.split(r"[:\-–—]\s*", base, maxsplit=1)[0].strip()

    n  = _extract_exp_num(base)
    rn = f"experiment {n}" if isinstance(n, int) else ""
    ex = f"exp {n}" if isinstance(n, int) else ""

    variants = {
        base,
        no_sub,
        _basic_norm(base),
        _basic_norm(no_sub),
        _slugify(base),
        _slugify(no_sub),
        rn, ex,
        rn.replace(" ", "_") if rn else "",
        ex.replace(" ", "_") if ex else "",
    }
    return sorted({v for v in variants if v})

# ---------- central alias index writer (append-only, idempotent) ----------

def _append_alias(project: Project, canonical_title: str, alias_title: str, step_name: str) -> None:
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
        "step": step_name,
        "experiment_title": canonical_title,
        "added_alias": alias_title,
    })

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- load/IO helpers ----------

def _load_table(full_path: Path, fmt: str, sheet: str | None) -> pd.DataFrame:
    if fmt == "csv":   return pd.read_csv(full_path, engine="python")
    if fmt == "tsv":   return pd.read_csv(full_path, sep="\t", engine="python")
    if fmt == "excel": return pd.read_excel(full_path, sheet_name=(sheet or 0))
    if fmt == "json":
        try:    return pd.read_json(full_path, lines=True)
        except: return pd.read_json(full_path)
    return pd.read_csv(full_path, engine="python")

def _infer_fmt_from_ext(relpath: str, fallback: str = "csv") -> str:
    ext = Path(relpath).suffix.lower()
    if ext == ".csv": return "csv"
    if ext in (".tsv", ".tab"): return "tsv"
    if ext in (".xlsx", ".xls"): return "excel"
    if ext == ".json": return "json"
    return fallback

def _safe_subject_token(x: Any) -> str:
    if x is None: return "unknown"
    s = str(x).strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s or "unknown"

def _write_subject_csv(dest_dir: Path, subject_id: str, df: pd.DataFrame) -> Tuple[str, int]:
    subject_id = _safe_subject_token(subject_id)
    out = dest_dir / f"subject_{subject_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        df.to_csv(out, index=False, encoding="utf-8")
    return str(out), int(len(df))

# ---------- step ----------

class ExecuteTrialSplit(BaseStep):
    """
    Materialize trial-wise subject CSVs per experiment based on meta/trial_split_plans.json.

    Output (combined): artifacts/meta/trials_index.json
      items[i] includes:
        - experiment_title
        - experiment_number (if detectable)
        - aliases []     # local variants for convenience
        - slug
        - dest_dir
        - subjects [...]
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="execute_trial_split",
            artifact="meta/trials_index.json",
            depends_on=[DescribeExperiments],
            map_over=DescribeExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        plans = project.artifacts_dir / "meta" / "trial_split_plans.json"
        out   = project.artifacts_dir / self.artifact
        return plans.exists() and (self._force or not out.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    # ---------- strategies (unchanged logic) ----------

    def _exec_per_subject(self, project: Project, plan: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
        src = plan.get("source", {}) if isinstance(plan, dict) else {}
        dir_rel   = src.get("dir", "")
        member_glob = src.get("member_glob", "*")
        fn_regex  = src.get("filename_id_regex", "")
        sid_from  = src.get("subject_id_from", "filename").lower()
        sid_col   = src.get("subject_id_column", "")

        root = project.artifacts_dir / "data_unpacked"
        src_dir = (root / dir_rel).resolve()
        results: List[Dict[str, Any]] = []
        errors: List[str] = []

        if not src_dir.exists() or not src_dir.is_dir():
            return {"subjects": [], "errors": [f"missing_dir:{dir_rel}"]}

        try:
            members = sorted([p for p in src_dir.glob(member_glob) if p.is_file()])
        except Exception as e:
            return {"subjects": [], "errors": [f"glob_error:{type(e).__name__}:{e}"]}

        pat = re.compile(fn_regex) if fn_regex else None

        for f in members:
            # subject id
            subj_id = f.stem
            if pat:
                m = pat.search(f.name)
                if m and m.group(1):
                    subj_id = m.group(1)
            # load table
            fmt = _infer_fmt_from_ext(f.name, fallback="csv")
            try:
                df = _load_table(f, fmt, sheet=None)
                if sid_from == "column" and sid_col and sid_col in df.columns:
                    vals = list(df[sid_col].dropna().unique().tolist())
                    if len(vals) == 1:
                        subj_id = str(vals[0])
                out_path, n_rows = _write_subject_csv(dest_dir, subj_id, df)
                results.append({"subject_id": _safe_subject_token(subj_id),
                                "relpath": str(Path(out_path).relative_to(project.artifacts_dir)),
                                "n_rows": n_rows})
            except Exception as e:
                errors.append(f"read_error:{f.name}:{type(e).__name__}:{e}")

        return {"subjects": results, "errors": errors}

    def _exec_per_experiment(self, project: Project, plan: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
        src = plan.get("source", {}) if isinstance(plan, dict) else {}
        relpath = src.get("relpath", "")
        sheet   = src.get("sheet", "") or None
        sid_col = src.get("subject_id_column", "")

        if not relpath or not sid_col:
            return {"subjects": [], "errors": [f"missing_relpath_or_subject_id_column"]}

        root = project.artifacts_dir / "data_unpacked"
        p = (root / relpath).resolve()
        if not p.exists():
            return {"subjects": [], "errors": [f"missing_file:{relpath}"]}

        fmt = _infer_fmt_from_ext(relpath, fallback="csv")
        try:
            df = _load_table(p, fmt, sheet)
        except Exception as e:
            return {"subjects": [], "errors": [f"load_error:{type(e).__name__}:{e}"]}

        if sid_col not in df.columns:
            return {"subjects": [], "errors": [f"missing_subject_id_column:{sid_col}"]}

        results: List[Dict[str, Any]] = []
        for sid, subdf in df.groupby(sid_col, dropna=False):
            out_path, n_rows = _write_subject_csv(dest_dir, sid, subdf)
            results.append({"subject_id": _safe_subject_token(sid),
                            "relpath": str(Path(out_path).relative_to(project.artifacts_dir)),
                            "n_rows": n_rows})

        return {"subjects": results, "errors": []}

    def _exec_one_file(self, project: Project, plan: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
        src = plan.get("source", {}) if isinstance(plan, dict) else {}
        relpath = src.get("relpath", "")
        sheet   = src.get("sheet", "") or None
        sid_col = src.get("subject_id_column", "")
        exp_col = src.get("experiment_column", "")
        exp_val = src.get("experiment_value", "")

        if not relpath or not sid_col or not exp_col:
            return {"subjects": [], "errors": [f"missing_relpath_or_columns"]}

        root = project.artifacts_dir / "data_unpacked"
        p = (root / relpath).resolve()
        if not p.exists():
            return {"subjects": [], "errors": [f"missing_file:{relpath}"]}

        fmt = _infer_fmt_from_ext(relpath, fallback="csv")
        try:
            df = _load_table(p, fmt, sheet)
        except Exception as e:
            return {"subjects": [], "errors": [f"load_error:{type(e).__name__}:{e}"]}

        if exp_col not in df.columns:
            return {"subjects": [], "errors": [f"missing_experiment_column:{exp_col}"]}
        if sid_col not in df.columns:
            return {"subjects": [], "errors": [f"missing_subject_id_column:{sid_col}"]}

        # Filter to the experiment value (string compare; best-effort)
        if exp_val != "":
            df = df[df[exp_col].astype(str) == str(exp_val)]

        results: List[Dict[str, Any]] = []
        for sid, subdf in df.groupby(sid_col, dropna=False):
            out_path, n_rows = _write_subject_csv(dest_dir, sid, subdf)
            results.append({"subject_id": _safe_subject_token(sid),
                            "relpath": str(Path(out_path).relative_to(project.artifacts_dir)),
                            "n_rows": n_rows})

        return {"subjects": results, "errors": []}

    # ---------- per-experiment ----------

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        art = project.artifacts_dir
        plans_path = art / "meta" / "trial_split_plans.json"
        try:
            plans = json.loads(plans_path.read_text(encoding="utf-8"))
        except Exception:
            plans = {}

        plan = {}
        if isinstance(plans.get("items"), list) and idx < len(plans["items"]):
            plan = plans["items"][idx]

        title = item.get("title") or f"Experiment {idx+1}"
        # If the plan uses a different title, record it as an alias (central index)
        plan_title = (plan.get("experiment_title") or "").strip()
        if plan_title and plan_title != title:
            _append_alias(project, canonical_title=title, alias_title=plan_title, step_name=self.name)

        slug  = _slugify(title)
        dest_dir = art / "trials" / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        out: Dict[str, Any] = {
            "experiment_title": title,
            "experiment_number": _extract_exp_num(title),
            "aliases": _alias_variants(title),  # local quality-of-life variants; central index is separate
            "slug": slug,
            "strategy": plan.get("strategy", "none"),
            "dest_dir": str(dest_dir.relative_to(project.artifacts_dir)),
            "subjects": [],
            "errors": [],
            "notes": [],
        }

        strategy = out["strategy"]
        if strategy == "per_subject":
            res = self._exec_per_subject(project, plan, dest_dir)
        elif strategy == "per_experiment":
            res = self._exec_per_experiment(project, plan, dest_dir)
        elif strategy == "one_file":
            res = self._exec_one_file(project, plan, dest_dir)
        else:
            out["notes"].append("no_action_for_strategy")
            return out

        out["subjects"] = res.get("subjects", [])
        out["errors"]   = res.get("errors", [])
        out["notes"].append(f"{len(out['subjects'])} subject files present/created")
        return out
