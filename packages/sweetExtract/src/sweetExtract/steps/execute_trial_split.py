# sweetExtract/steps/execute_trial_split.py
from __future__ import annotations
import json, re, unicodedata, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

_ROMAN = {'M':1000,'CM':900,'D':500,'CD':400,'C':100,'XC':90,'L':50,'XL':40,'X':10,'IX':9,'V':5,'IV':4,'I':1}
_STOP  = {"experiment","experiments","exp","study","studies"}

def _roman_to_int(s: str) -> int | None:
    s = s.upper(); i = 0; n = 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in _ROMAN: n += _ROMAN[s[i:i+2]]; i += 2
        elif s[i] in _ROMAN: n += _ROMAN[s[i]]; i += 1
        else: return None
    return n

def _extract_exp_num(title: str) -> int | None:
    m = re.search(r"\b(?:experiment|exp)\s*([0-9ivxlcdm]+)\b", title or "", flags=re.I)
    if not m: return None
    g = m.group(1); return int(g) if g.isdigit() else _roman_to_int(g)

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
    title = title or ""; base = title.strip()
    no_sub = re.split(r"[:\-–—]\s*", base, maxsplit=1)[0].strip()
    n = _extract_exp_num(base)
    rn = f"experiment {n}" if isinstance(n, int) else ""
    ex = f"exp {n}" if isinstance(n, int) else ""
    variants = {
        base, no_sub, _basic_norm(base), _basic_norm(no_sub), _slugify(base), _slugify(no_sub),
        rn, ex, rn.replace(" ","_") if rn else "", ex.replace(" ","_") if ex else ""
    }
    return sorted({v for v in variants if v})

def _append_alias(project: Project, canonical_title: str, alias_title: str, step_name: str) -> None:
    alias_title = (alias_title or "").strip(); canonical_title = (canonical_title or "").strip()
    if not alias_title or alias_title == canonical_title: return
    path = project.artifacts_dir / "meta" / "experiment_aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"items": []}
    if path.exists():
        try: data = json.loads(path.read_text(encoding="utf-8")) or {"items": []}
        except Exception: data = {"items": []}
    items = data.get("items");
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
    updates.append({"ts": datetime.datetime.utcnow().isoformat()+"Z","step": step_name,
                    "experiment_title": canonical_title,"added_alias": alias_title})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

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
    if ext in (".tsv",".tab"): return "tsv"
    if ext in (".xlsx",".xls"): return "excel"
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
    # ALWAYS overwrite to avoid stale leftovers across runs
    df.to_csv(out, index=False, encoding="utf-8")
    return str(out), int(len(df))

def _purge_subject_outputs(dest_dir: Path) -> None:
    """Remove old subject csvs before writing new outputs (prevents mixing across runs/strategies)."""
    if not dest_dir.exists():
        return
    for p in dest_dir.glob("subject_*.csv"):
        try:
            p.unlink()
        except Exception:
            pass

# -------- robust row filters + audit --------

_ALLOWED_OPS = {"==","!=", "in","not_in","contains","regex",">",">=","<","<="}

def _is_numeric_like(series: pd.Series) -> bool:
    try:
        _ = pd.to_numeric(series, errors="coerce")
        return True
    except Exception:
        return False

def _parse_number(s: str):
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return None

def _normalize_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def _apply_row_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Robust filtering:
      - For ==, != : numeric compare when possible; else case-insensitive string compare.
      - For in, not_in: supports mixed numeric/string lists.
      - contains/regex: string-based.
      - >, >=, <, <= : numeric compare.
    Adds an audit list to df.attrs['_filter_audit'] with rows_before/rows_after per filter.
    """
    if not isinstance(filters, list) or not filters:
        return df

    out = df
    for f in filters:
        try:
            col = f.get("column")
            op  = f.get("op")
            vstr = str(f.get("value_str", ""))
            vals = f.get("values") or []

            if not col or op not in _ALLOWED_OPS or col not in out.columns:
                continue

            col_series = out[col]
            str_series = _normalize_str_series(col_series)
            num_series = pd.to_numeric(col_series, errors="coerce")

            before = len(out)

            if op in {"==", "!="}:
                vnum = _parse_number(vstr)
                if vnum is not None and _is_numeric_like(col_series):
                    mask = (num_series == vnum)
                else:
                    mask = (str_series == vstr.strip().lower())
                if op == "!=":
                    mask = ~mask
                out = out[mask]

            elif op in {"in", "not_in"}:
                num_vals, str_vals = [], []
                for v in vals:
                    vnum = _parse_number(str(v))
                    if vnum is not None:
                        num_vals.append(vnum)
                    else:
                        str_vals.append(str(v).strip().lower())
                if num_vals and _is_numeric_like(col_series):
                    mask_num = num_series.isin(num_vals)
                else:
                    mask_num = pd.Series(False, index=out.index)
                mask_str = str_series.isin(str_vals) if str_vals else pd.Series(False, index=out.index)
                mask = mask_num | mask_str
                if op == "not_in":
                    mask = ~mask
                out = out[mask]

            elif op == "contains":
                out = out[str_series.str.contains(vstr.strip().lower(), na=False, regex=False)]

            elif op == "regex":
                out = out[col_series.astype(str).str.contains(vstr, na=False, regex=True)]

            elif op in {">", ">=", "<", "<="}:
                try:
                    vnum = float(vstr)
                except Exception:
                    continue
                s_num = pd.to_numeric(col_series, errors="coerce")
                if op == ">":   out = out[s_num >  vnum]
                if op == ">=":  out = out[s_num >= vnum]
                if op == "<":   out = out[s_num <  vnum]
                if op == "<=":  out = out[s_num <= vnum]

            out.attrs.setdefault("_filter_audit", []).append({
                "column": col, "op": op, "value_str": vstr, "values": vals,
                "rows_before": int(before), "rows_after": int(len(out))
            })
        except Exception:
            continue
    return out

# ---------- step ----------

class ExecuteTrialSplit(BaseStep):
    """
    Materialize trial-wise subject CSVs per experiment based on meta/trial_split_plans.json.

    Plan 'source' may include optional 'row_filters' (list of {column, op, value_str, values})
    applied after loading each table and before grouping/saving.
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="execute_trial_split",
            artifact="meta/trials_index.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        plans = project.artifacts_dir / "meta" / "trial_split_plans.json"
        out   = project.artifacts_dir / self.artifact
        return plans.exists() and (self._force or not out.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    # ---------- strategies ----------

    def _exec_per_subject(self, project: Project, plan: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
        src = plan.get("source", {}) if isinstance(plan, dict) else {}
        dir_rel     = src.get("dir", "")
        member_glob = src.get("member_glob", "*")
        fn_regex    = src.get("filename_id_regex", "")
        sid_from    = src.get("subject_id_from", "filename").lower()
        sid_col     = src.get("subject_id_column", "")
        row_filters = src.get("row_filters") or []

        root = project.artifacts_dir / "data_unpacked"
        src_dir = (root / dir_rel).resolve()
        results: List[Dict[str, Any]] = []; errors: List[str] = []
        filter_audit_all: List[Dict[str, Any]] = []

        if not src_dir.exists() or not src_dir.is_dir():
            return {"subjects": [], "errors": [f"missing_dir:{dir_rel}"], "filter_audit": []}

        try:
            members = sorted([p for p in src_dir.glob(member_glob) if p.is_file()])
        except Exception as e:
            return {"subjects": [], "errors": [f"glob_error:{type(e).__name__}:{e}"], "filter_audit": []}

        pat = re.compile(fn_regex) if fn_regex else None

        for f in members:
            subj_id = f.stem
            if pat:
                m = pat.search(f.name)
                if m:
                    subj_id = m.groupdict().get("subject_id") or (m.group(1) if m.groups() else f.stem)

            fmt = _infer_fmt_from_ext(f.name, fallback="csv")
            try:
                df = _load_table(f, fmt, sheet=None)

                if sid_from == "column" and sid_col and sid_col in df.columns:
                    vals = list(df[sid_col].dropna().unique().tolist())
                    if len(vals) == 1:
                        subj_id = str(vals[0])

                df = _apply_row_filters(df, row_filters)
                audit = df.attrs.pop("_filter_audit", [])
                for a in audit:
                    a.update({"file": f.name})
                filter_audit_all.extend(audit)

                if sid_from == "column" and sid_col and sid_col in df.columns:
                    df = df.drop(columns=[sid_col], errors="ignore")

                out_path, n_rows = _write_subject_csv(dest_dir, subj_id, df)
                results.append({
                    "subject_id": _safe_subject_token(subj_id),
                    "relpath": str(Path(out_path).relative_to(project.artifacts_dir)),
                    "n_rows": n_rows
                })
            except Exception as e:
                errors.append(f"read_error:{f.name}:{type(e).__name__}:{e}")

        return {"subjects": results, "errors": errors, "filter_audit": filter_audit_all}

    def _exec_per_experiment(self, project: Project, plan: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
        src = plan.get("source", {}) if isinstance(plan, dict) else {}
        relpath = src.get("relpath", ""); sheet = src.get("sheet", "") or None
        sid_col = src.get("subject_id_column", "")
        row_filters = src.get("row_filters") or []

        if not relpath or not sid_col:
            return {"subjects": [], "errors": ["missing_relpath_or_subject_id_column"], "filter_audit": []}

        root = project.artifacts_dir / "data_unpacked"
        p = (root / relpath).resolve()
        if not p.exists():
            return {"subjects": [], "errors": [f"missing_file:{relpath}"], "filter_audit": []}

        fmt = _infer_fmt_from_ext(relpath, fallback="csv")
        try:
            df = _load_table(p, fmt, sheet)
        except Exception as e:
            return {"subjects": [], "errors": [f"load_error:{type(e).__name__}:{e}"], "filter_audit": []}

        if sid_col not in df.columns:
            return {"subjects": [], "errors": [f"missing_subject_id_column:{sid_col}"], "filter_audit": []}

        df = _apply_row_filters(df, row_filters)
        filter_audit = df.attrs.pop("_filter_audit", [])

        results: List[Dict[str, Any]] = []
        for sid, subdf in df.groupby(sid_col, dropna=False):
            subdf2 = subdf.drop(columns=[sid_col], errors="ignore")
            out_path, n_rows = _write_subject_csv(dest_dir, sid, subdf2)
            results.append({
                "subject_id": _safe_subject_token(sid),
                "relpath": str(Path(out_path).relative_to(project.artifacts_dir)),
                "n_rows": n_rows
            })

        return {"subjects": results, "errors": [], "filter_audit": filter_audit}

    def _exec_one_file(self, project: Project, plan: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
        src = plan.get("source", {}) if isinstance(plan, dict) else {}
        relpath = src.get("relpath", ""); sheet = src.get("sheet", "") or None
        sid_col = src.get("subject_id_column", "")
        row_filters = src.get("row_filters") or []

        if not relpath or not sid_col:
            return {"subjects": [], "errors": ["missing_relpath_or_subject_id_column"], "filter_audit": []}

        root = project.artifacts_dir / "data_unpacked"
        p = (root / relpath).resolve()
        if not p.exists():
            return {"subjects": [], "errors": [f"missing_file:{relpath}"], "filter_audit": []}

        fmt = _infer_fmt_from_ext(relpath, fallback="csv")
        try:
            df = _load_table(p, fmt, sheet)
        except Exception as e:
            return {"subjects": [], "errors": [f"load_error:{type(e).__name__}:{e}"], "filter_audit": []}

        df = _apply_row_filters(df, row_filters)
        filter_audit = df.attrs.pop("_filter_audit", [])

        if sid_col not in df.columns:
            return {"subjects": [], "errors": [f"missing_subject_id_column:{sid_col}"], "filter_audit": filter_audit}

        results: List[Dict[str, Any]] = []
        for sid, subdf in df.groupby(sid_col, dropna=False):
            subdf2 = subdf.drop(columns=[sid_col], errors="ignore")
            out_path, n_rows = _write_subject_csv(dest_dir, sid, subdf2)
            results.append({
                "subject_id": _safe_subject_token(sid),
                "relpath": str(Path(out_path).relative_to(project.artifacts_dir)),
                "n_rows": n_rows
            })
        return {"subjects": results, "errors": [], "filter_audit": filter_audit}

    # ---------- mapping ----------

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        art = project.artifacts_dir
        plans_path = art / "meta" / "trial_split_plans.json"
        try: plans = json.loads(plans_path.read_text(encoding="utf-8"))
        except Exception: plans = {}

        plan = {}
        if isinstance(plans.get("items"), list) and idx < len(plans["items"]):
            plan = plans["items"][idx]

        title = item.get("title") or f"Experiment {idx+1}"
        plan_title = (plan.get("experiment_title") or "").strip()
        if plan_title and plan_title != title:
            _append_alias(project, canonical_title=title, alias_title=plan_title, step_name=self.name)

        slug  = _slugify(title)
        dest_dir = art / "trials" / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        # PURGE old subject outputs to avoid mixing across strategies/filters
        _purge_subject_outputs(dest_dir)

        out: Dict[str, Any] = {
            "experiment_title": title,
            "experiment_number": _extract_exp_num(title),
            "aliases": _alias_variants(title),
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
            manifest = {
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "strategy": strategy,
                "source": plan.get("source", {}),
                "row_filters": (plan.get("source", {}) or {}).get("row_filters", []),
                "filter_audit": [],
                "n_subjects": 0,
                "n_errors": 0,
                "output_files": [],
            }
            # write manifests (dest_dir + meta)
            # (dest_dir / "_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            meta_dir = art / "meta" / "trial_split_runs"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"{idx:03d}_{slug}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return out

        out["subjects"] = res.get("subjects", [])
        out["errors"]   = res.get("errors", [])
        out["notes"].append(f"{len(out['subjects'])} subject files present/created")

        # tiny run manifest for validator (includes row_filters + filter_audit + outputs)
        manifest = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "strategy": strategy,
            "source": plan.get("source", {}),
            "row_filters": (plan.get("source", {}) or {}).get("row_filters", []),
            "filter_audit": res.get("filter_audit", []),
            "n_subjects": len(out["subjects"]),
            "n_errors": len(out["errors"]),
            "output_files": [s.get("relpath","") for s in out["subjects"]],
        }
        # write manifests (dest_dir + meta)
        (dest_dir / "_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_dir = art / "meta" / "trial_split_runs"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"{idx:03d}_{slug}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
