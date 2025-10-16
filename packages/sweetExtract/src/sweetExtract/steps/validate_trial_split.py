# sweetExtract/steps/validate_trial_split.py
from __future__ import annotations
import os, json, statistics, datetime, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.execute_trial_split import ExecuteTrialSplit

# ---------- utils ----------

def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _safe_glob_subject_csvs(project: Project, idx_block: Dict[str, Any]) -> List[Path]:
    dest_rel = (idx_block or {}).get("dest_dir") or ""
    if not dest_rel:
        return []
    base = (project.artifacts_dir / dest_rel).resolve()
    if not base.exists():
        return []
    return sorted([p for p in base.glob("subject_*.csv") if p.is_file()])

def _summarize_subject_files(files: List[Path]) -> Tuple[int, int, Dict[str, int], List[str]]:
    n_total = 0
    per_file: Dict[str, int] = {}
    ids: List[str] = []
    for p in files:
        try:
            df = pd.read_csv(p, engine="python")
            n = int(len(df))
            n_total += n
            per_file[p.name] = n
            if len(ids) < 6:
                ids.append(p.stem.replace("subject_", "", 1))
        except Exception:
            per_file[p.name] = -1
    return len(files), n_total, per_file, ids

def _per_subject_stats(per_file_counts: Dict[str, int]) -> Dict[str, int]:
    vals = [v for v in per_file_counts.values() if isinstance(v, int) and v >= 0]
    if not vals:
        return {"min": -1, "max": -1, "median": -1}
    return {"min": int(min(vals)), "max": int(max(vals)), "median": int(statistics.median(vals))}

def _read_plan_for_idx(project: Project, idx: int) -> Dict[str, Any]:
    plans = _read_json(project.artifacts_dir / "meta" / "trial_split_plans.json") or {}
    it = {}
    if isinstance(plans.get("items"), list) and idx < len(plans["items"]):
        it = plans["items"][idx]
    src = (it or {}).get("source") or {}
    return {
        "strategy": (it or {}).get("strategy") or "",
        "source": {
            "relpath": src.get("relpath", "") or "",
            "sheet": src.get("sheet", "") or "",
            "dir": src.get("dir", "") or "",
            "member_glob": src.get("member_glob", "") or "",
            "filename_id_regex": src.get("filename_id_regex", "") or "",
            "subject_id_from": src.get("subject_id_from", "") or "",
            "subject_id_column": src.get("subject_id_column", "") or "",
            "experiment_column": src.get("experiment_column", "") or "",
            "experiment_value": src.get("experiment_value", "") or "",
            "row_filters": src.get("row_filters") or [],
        },
    }

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s or "exp"

def _read_manifest_for_idx(project: Project, idx: int, exp_title: str) -> Dict[str, Any]:
    art = project.artifacts_dir
    slug = _slugify(exp_title)
    meta_manifest = art / "meta" / "trial_split_runs" / f"{idx:03d}_{slug}.json"
    dir_manifest  = art / "trials" / slug / "_run_manifest.json"
    m = _read_json(meta_manifest) or _read_json(dir_manifest) or {}
    # normalize shape for prompt (never crash)
    m.setdefault("strategy", "")
    m.setdefault("source", {})
    m.setdefault("row_filters", [])
    m.setdefault("filter_audit", [])
    m.setdefault("n_subjects", 0)
    m.setdefault("n_errors", 0)
    return m

def _load_probes_lines(project: Project, idx: int) -> List[str]:
    """
    Pull short, high-signal lines from candidate probes.
    This matches what the refiner sees: headers + ID-uniqueness sketches.
    """
    probes_all = _read_json(project.artifacts_dir / "meta" / "candidate_probes.json") or {}
    block = None
    if isinstance(probes_all.get("items"), list) and idx < len(probes_all["items"]):
        block = probes_all["items"][idx]
    lines: List[str] = []
    for p in (block or {}).get("probes", [])[:6]:
        rp = p.get("relpath", ""); sh = p.get("sheet", ""); fmt = p.get("format", "")
        cols = ", ".join((p.get("columns") or [])[:12])
        subj_cols = p.get("present_subject_cols") or []
        trial_cols = p.get("present_trial_cols") or []
        exp_cols = p.get("present_experiment_cols") or []
        stat = p.get("stats") or {}; er = stat.get("error", ""); nr = stat.get("n_rows", -1); nc = stat.get("n_cols", 0)

        subj_u = p.get("subject_uniques") or {}
        exp_u  = p.get("experiment_uniques") or {}

        def _fmt_u(d: Dict[str, Any]) -> str:
            bits = []
            for k, v in d.items():
                cnt = v.get("unique_count", 0)
                ex  = v.get("examples", [])
                # keep examples compact; they were already deduped at probe time
                ex_str = f" ex=[{', '.join(map(str, ex))}]" if ex else ""
                bits.append(f"{k}: count={cnt}{ex_str}")
            return "; ".join(bits) if bits else "(none)"

        lines.append(
            f"- relpath={rp} | sheet={sh} | fmt={fmt} | n_rows~{nr} n_cols~{nc} err={er} "
            f"| subj_cols={subj_cols} uniques: {_fmt_u(subj_u)} "
            f"| trial_cols={trial_cols} "
            f"| exp_cols={exp_cols} uniques: {_fmt_u(exp_u)} "
            f"| cols=[{cols}]"
        )
    return lines

def _expected_subjects_from_item(item: Dict[str, Any]) -> int | None:
    try:
        n = (((item or {}).get("decision") or {}).get("participant_n_reported"))
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:
        return None

def _format_plan_lines(plan: Dict[str, Any]) -> List[str]:
    st = plan.get("strategy", "")
    src = plan.get("source") or {}
    rf  = src.get("row_filters") or []
    lines = [
        f"- strategy={st}",
        f"- source.relpath={src.get('relpath','')} sheet={src.get('sheet','')}",
        f"- source.dir={src.get('dir','')} member_glob={src.get('member_glob','')}",
        f"- subject_id_from={src.get('subject_id_from','')} subject_id_column={src.get('subject_id_column','')}",
    ]
    if rf:
        for f in rf[:6]:
            lines.append(f"- row_filter: column={f.get('column','')} op={f.get('op','')} value_str={f.get('value_str','')} values={f.get('values') or []}")
    return lines

def _format_manifest_lines(m: Dict[str, Any]) -> List[str]:
    lines = [
        f"- executed.strategy={m.get('strategy','')}",
        f"- executed.row_filters={m.get('row_filters', [])}",
        f"- executed.n_subjects={m.get('n_subjects', 0)} executed.n_errors={m.get('n_errors', 0)}",
    ]
    fa = m.get("filter_audit") or []
    for a in fa[:8]:
        lines.append(
            f"- filter_audit: col={a.get('column','')} op={a.get('op','')} value_str={a.get('value_str','')} "
            f"values={a.get('values') or []} rows_before~{a.get('rows_before',-1)} rows_after~{a.get('rows_after',-1)} file={a.get('file','')}"
        )
    return lines

# ---------- LLM glue ----------

def _generate(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        # Fallback minimal object (keeps pipeline alive without LLM)
        return {
            "experiment_title": "",
            "validated": False,
            "reasons": ["LLM unavailable"],
            "hint_text": "Refine strategy using generic guidance; consider row filters or filename/sheet cues.",
            "hint": {
                "previous_plan": {"strategy": "none", "source": {
                    "relpath": "", "sheet": "", "dir": "", "member_glob": "",
                    "filename_id_regex": "", "subject_id_from": "", "subject_id_column": "",
                    "experiment_column": "", "experiment_value": "", "row_filters": []
                }},
                "observed": {
                    "n_subject_files": 0, "n_rows_total": 0,
                    "per_subject_rows": {"min": -1, "max": -1, "median": -1},
                    "subject_ids_sample": [], "errors": []
                },
                "recommendations": [{
                    "type": "strategy_review", "reason": "LLM unavailable",
                    "options": ["per_subject","per_experiment","one_file"],
                    "strategy_to_try": None, "candidate_columns": None,
                    "suggested_filename_id_regex": None, "suggested_subject_id_from": None,
                    "dir": None, "member_glob": None, "relpath": None, "sheet": None,
                    "subject_id_column": None, "experiment_column": None, "experiment_value": None
                }]
            },
            "confidence": 0.0,
            "model": "validator-fallback"
        }

    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You are a careful data validator. Determine whether the produced subject–trial CSV split "
            "correctly isolates the TARGET experiment.\n"
            "You will receive: (1) experiment description; (2) previous plan; (3) verified probe facts "
            "(headers + ID uniqueness counts + short unique examples); (4) execution manifest (row filter audit, counts).\n\n"
            "Rules:\n"
            "1) Be conservative. Validate only if evidence strongly supports the split.\n"
            "2) Use probe uniqueness and the filter audit to reason about mixing (e.g., exp/condition columns with >1 unique).\n"
            "3) Recommend generic improvements that are broadly applicable (e.g., consider row_filters on plausible ID/condition columns; "
            "   consider filename-based subject parsing; consider member glob narrowing). Do not depend on private file paths beyond relpath/dir.\n"
            "4) Return ONLY JSON matching the schema."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="TrialSplitValidation",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return out or {}

# ---------- step ----------

class ValidateTrialSplit(BaseStep):
    """
    LLM-backed validation that emits generic, implementable recommendations.
    Consumes: ExecuteTrialSplit outputs, candidate probes, and previous plan.
    Produces: artifacts/validate_trial_split/{idx}.json and meta/validate_trial_split.json
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="validate_trial_split",
            artifact="meta/validate_trial_split.json",
            depends_on=[FilterEmpiricalExperiments, ExecuteTrialSplit],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        # always run; BaseStep will skip if combined artifact already exists unless forcing
        return True

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        art = project.artifacts_dir

        # 1) Gather inputs
        title = item.get("title") or f"Experiment {idx+1}"
        desc  = (item.get("standalone_summary") or "").strip()

        # trials index (what ExecuteTrialSplit returned for this idx)
        trials_all = _read_json(art / "meta" / "trials_index.json") or {}
        trials_block = ((trials_all.get("items") or [None]*9999)[idx]) if isinstance(trials_all, dict) else {}

        # subject CSVs summary
        subject_files = _safe_glob_subject_csvs(project, trials_block)
        n_files, n_rows_total, per_file_counts, ids_sample = _summarize_subject_files(subject_files)
        per_stats = _per_subject_stats(per_file_counts)
        errors_exec = (trials_block or {}).get("errors") or []

        # previous plan
        previous_plan = _read_plan_for_idx(project, idx)
        plan_lines = _format_plan_lines(previous_plan)

        # probes + uniqueness sketches
        probe_lines = _load_probes_lines(project, idx)

        # manifest from execute step (with filter audit)
        manifest = _read_manifest_for_idx(project, idx, title)
        manifest_lines = _format_manifest_lines(manifest)

        # expected subject count (if available in meta for this experiment)
        expected_n = _expected_subjects_from_item(item)
        exp_n_line = f"EXPECTED_N_SUBJECTS: ~{expected_n}" if isinstance(expected_n, int) and expected_n > 0 else "EXPECTED_N_SUBJECTS: (unknown)"

        # 2) Build prompt
        parts: List[str] = []
        parts.append(f"TARGET EXPERIMENT (canonical): {title}\n")
        parts.append(exp_n_line + "\n")
        parts.append("DESCRIPTION:\n" + (desc or "(none)") + "\n")
        parts.append("PREVIOUS PLAN:\n" + ("\n".join(plan_lines) if plan_lines else "(none)") + "\n")
        parts.append("PROBES (verified facts — includes ID-column uniqueness sketches):\n" + ("\n".join(probe_lines) if probe_lines else "(none)") + "\n")
        parts.append("EXECUTION MANIFEST:\n" + ("\n".join(manifest_lines) if manifest_lines else "(none)") + "\n")
        parts.append("SUBJECT FILES SUMMARY:\n" +
                     f"- n_subject_files={n_files}\n"
                     f"- per_subject_rows: min={per_stats['min']} max={per_stats['max']} median={per_stats['median']}\n"
                     f"- subject_ids_sample={ids_sample}\n"
                     f"- exec_errors={errors_exec}\n")
        parts.append(
            "TASK:\n"
            "- Decide validated=true/false (be conservative).\n"
            "- Provide reasons (bullet-like short sentences).\n"
            "- Provide a brief hint_text.\n"
            "- Provide a 'hint' object:\n"
            "    previous_plan: echo strategy and source (incl. row_filters) for reproducibility.\n"
            "    observed: n_subject_files, n_rows_total, per_subject_rows{min,max,median}, subject_ids_sample, errors.\n"
            "    recommendations: generic, implementable suggestions.\n"
            "      Valid types: strategy_review, consider_row_filter, consider_filename_id_regex, consider_member_glob.\n"
            "- Return ONLY JSON matching the schema."
        )
        prompt = "\n".join(parts)

        # 3) Strict schema for LLM output (responses.parse requires 'required' list to include ALL property keys)
        rec_item_props = {
            "type": {"type": "string"},
            "reason": {"type": "string"},
            "options": {"type": ["array", "null"], "items": {"type": "string"}},
            "strategy_to_try": {"type": ["string", "null"]},
            "candidate_columns": {"type": ["array", "null"], "items": {"type": "string"}},
            "suggested_filename_id_regex": {"type": ["string", "null"]},
            "suggested_subject_id_from": {"type": ["string", "null"]},
            "dir": {"type": ["string", "null"]},
            "member_glob": {"type": ["string", "null"]},
            "relpath": {"type": ["string", "null"]},
            "sheet": {"type": ["string", "null"]},
            "subject_id_column": {"type": ["string", "null"]},
            "experiment_column": {"type": ["string", "null"]},
            "experiment_value": {"type": ["string", "null"]},
        }
        rec_item_required = list(rec_item_props.keys())

        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experiment_title": {"type": "string"},
                "validated": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "hint_text": {"type": "string"},
                "hint": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "previous_plan": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "strategy": {"type": "string"},
                                "source": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "relpath": {"type": "string"},
                                        "sheet": {"type": "string"},
                                        "dir": {"type": "string"},
                                        "member_glob": {"type": "string"},
                                        "filename_id_regex": {"type": "string"},
                                        "subject_id_from": {"type": "string"},
                                        "subject_id_column": {"type": "string"},
                                        "experiment_column": {"type": "string"},
                                        "experiment_value": {"type": "string"},
                                        "row_filters": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "column": {"type": "string"},
                                                    "op": {"type": "string"},
                                                    "value_str": {"type": "string"},
                                                    "values": {"type": "array", "items": {"type": "string"}},
                                                },
                                                "required": ["column", "op", "value_str", "values"],
                                            },
                                        },
                                    },
                                    "required": [
                                        "relpath", "sheet", "dir", "member_glob", "filename_id_regex",
                                        "subject_id_from", "subject_id_column", "experiment_column",
                                        "experiment_value", "row_filters",
                                    ],
                                },
                            },
                            "required": ["strategy", "source"],
                        },
                        "observed": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "n_subject_files": {"type": "integer"},
                                "n_rows_total": {"type": "integer"},
                                "per_subject_rows": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "min": {"type": "integer"},
                                        "max": {"type": "integer"},
                                        "median": {"type": "integer"},
                                    },
                                    "required": ["min", "max", "median"],
                                },
                                "subject_ids_sample": {"type": "array", "items": {"type": "string"}},
                                "errors": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["n_subject_files", "n_rows_total", "per_subject_rows", "subject_ids_sample", "errors"],
                        },
                        "recommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": rec_item_props,
                                "required": rec_item_required,
                            },
                        },
                    },
                    "required": ["previous_plan", "observed", "recommendations"],
                },
                "confidence": {"type": "number"},
                "model": {"type": "string"},
            },
            "required": ["experiment_title", "validated", "reasons", "hint_text", "hint", "confidence", "model"],
        }

        # 4) Call LLM
        out = _generate(prompt, schema)
        if not isinstance(out, dict):
            out = {}

        # Ensure safe defaults + canonical title
        out.setdefault("experiment_title", title)
        out.setdefault("validated", False)
        out.setdefault("reasons", ["Needs refinement or confirmation."])
        out.setdefault("hint_text", "Refine strategy using generic guidance; you may incorporate row filters or filename/sheet cues.")
        out.setdefault("hint", {
            "previous_plan": previous_plan,
            "observed": {
                "n_subject_files": n_files,
                "n_rows_total": n_rows_total,
                "per_subject_rows": per_stats,
                "subject_ids_sample": ids_sample,
                "errors": errors_exec
            },
            "recommendations": [{
                "type": "strategy_review", "reason": "Add generic checks.",
                "options": ["per_subject","per_experiment","one_file"],
                "strategy_to_try": None, "candidate_columns": None,
                "suggested_filename_id_regex": None, "suggested_subject_id_from": None,
                "dir": None, "member_glob": None, "relpath": None, "sheet": None,
                "subject_id_column": None, "experiment_column": None, "experiment_value": None
            }]
        })
        out.setdefault("confidence", 0.3)
        out.setdefault("model", "validator")

        # 5) Write per-item artifact for downstream visibility
        per_item_path = art / "validate_trial_split" / f"{idx}.json"
        _write_json(per_item_path, out)
        return out
