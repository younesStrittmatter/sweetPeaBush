from __future__ import annotations
import os, json, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

# ---------- io helpers ----------

def _read_json(p: Path) -> Any:
    if not p.exists(): return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

# ---------- prompt formatters ----------

def _lines_from_candidates(cand_block: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for c in (cand_block.get("candidates") or [])[:4]:
        rp = c.get("relpath",""); sh = c.get("sheet",""); fmt = c.get("format","")
        sid = ", ".join((c.get("subject_id_hints") or [])[:2])
        tid = ", ".join((c.get("trial_index_hints") or [])[:2])
        lines.append(f"- relpath={rp} | sheet={sh} | format={fmt} | subject_hints=[{sid}] | trial_hints=[{tid}]")
    return lines

def _fmt_uniques(one: Dict[str, Any]) -> str:
    bits = []
    for col, stats in (one or {}).items():
        cnt = stats.get("unique_count", 0)
        ex  = stats.get("examples") or []
        trunc = bool(stats.get("truncated"))
        src = stats.get("source") or "sample"
        ex_str = ", ".join([str(x) for x in ex])
        if trunc and ex:
            bits.append(f"{col}: count={cnt}, ex=[{ex_str}] (truncated) src={src}")
        elif ex:
            bits.append(f"{col}: count={cnt}, ex=[{ex_str}] src={src}")
        else:
            bits.append(f"{col}: count={cnt} src={src}")
    return "; ".join(bits)

def _lines_from_probes(probe_block: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for p in (probe_block.get("probes") or [])[:6]:
        rp = p.get("relpath",""); sh = p.get("sheet",""); fmt = p.get("format","")
        cols = ", ".join((p.get("columns") or [])[:12])
        subj = ", ".join((p.get("present_subject_cols") or []))
        tri  = ", ".join((p.get("present_trial_cols") or []))
        exps = ", ".join((p.get("present_experiment_cols") or []))
        stat = p.get("stats") or {}
        er   = stat.get("error",""); nr = stat.get("n_rows", -1); nc = stat.get("n_cols", 0)
        subj_u = _fmt_uniques(p.get("subject_uniques") or {})
        exp_u  = _fmt_uniques(p.get("experiment_uniques") or {})
        lines.append(
            f"- relpath={rp} | sheet={sh} | fmt={fmt} | n_rows~{nr} n_cols~{nc} err={er} "
            f"| subj_cols=[{subj}] uniques: {subj_u or '—'} | trial_cols=[{tri}] "
            f"| exp_cols=[{exps}] uniques: {exp_u or '—'} | cols=[{cols}]"
        )
    return lines

# ---------- validation feedback helpers ----------

def _read_validation_hint_for_idx(project: Project, idx: int) -> Tuple[str, Dict[str, Any]]:
    p = project.artifacts_dir / "validate_trial_split" / f"{idx}.json"
    obj = _read_json(p) or {}
    hint_text = (obj.get("hint_text") or "").strip()
    hint_struct = obj.get("hint") or {}
    if isinstance(hint_struct, dict):
        if "previous_plan" not in hint_struct and "last_plan" in hint_struct:
            hint_struct["previous_plan"] = hint_struct.get("last_plan")
    return hint_text, (hint_struct if isinstance(hint_struct, dict) else {})

def _format_hint_struct(hint_struct: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    recs = hint_struct.get("recommendations") or []
    for rec in recs:
        rtype = (rec.get("type") or "").strip()
        reason = (rec.get("reason") or "").strip()
        if rtype in ("switch_strategy","strategy_switch"):
            to = rec.get("strategy_to_try") or ""
            lines.append(f"- consider_switch_strategy_to={to}  # {reason}")
        elif rtype in ("consider_row_filter","row_filter"):
            cols = ", ".join((rec.get("candidate_columns") or [])[:6])
            lines.append(f"- consider_row_filter on [{cols}]  # {reason}")
        elif rtype in ("consider_filename_id_regex","filename_regex"):
            lines.append(f"- consider_filename_id_regex  # {reason}")
        elif rtype in ("consider_member_glob","dir_member_glob","dir_member_glob_candidates"):
            lines.append(f"- consider_member_glob  # {reason}")
        elif rtype == "strategy_review":
            opts = ", ".join(rec.get("options") or [])
            lines.append(f"- strategy_review: {reason}; options=[{opts}]")
        elif rtype == "check_sources":
            lines.append(f"- check_sources: {reason}")
        elif reason:
            lines.append(f"- {rtype}: {reason}")

    prev = hint_struct.get("previous_plan") or {}
    if isinstance(prev, dict):
        st = prev.get("strategy",""); src = prev.get("source") or {}
        if st:
            lines.append(f"- previous_plan: strategy={st} source_keys={list(src.keys())}")
    return lines

def _expected_subjects_from_filter_item(filter_item: Dict[str, Any]) -> int | None:
    try:
        n = (((filter_item or {}).get("decision") or {}).get("participant_n_reported"))
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:
        return None

# ---------- LLM wrapper ----------

def _generate(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {
            "experiment_title": "", "strategy": "none", "reason": "LLM unavailable",
            "source": {
                "relpath": "", "sheet": "", "dir": "", "member_glob": "", "filename_id_regex": "",
                "subject_id_from": "column", "subject_id_column": "", "experiment_column": "", "experiment_value": "",
                "row_filters": []
            },
            "diagnostics": {"notes": ["LLM unavailable"], "confidence": 0.0, "model": ""}
        }
    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "Refine a plan to produce subject–trial-wise CSVs for ONE experiment.\n"
            "Inputs: description, candidates, verified probe facts (headers + uniqueness counts/examples with source), and validation feedback.\n"
            "Pick EXACTLY ONE strategy (per_subject / per_experiment / one_file).\n"
            "You MAY include generic row_filters (column/op/value(s)) to isolate the target experiment/condition when appropriate.\n"
            "Do NOT rename or drop columns beyond filtering. Use ONLY columns/values seen in PROBES."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="TrialSplitPlanRefined",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return out or {}

# ---------- alias writer ----------

def _append_alias(project: Project, canonical_title: str, alias_title: str, source_step: str) -> None:
    alias_title = (alias_title or "").strip(); canonical_title = (canonical_title or "").strip()
    if not alias_title or alias_title == canonical_title: return
    path = project.artifacts_dir / "meta" / "experiment_aliases.json"; path.parent.mkdir(parents=True, exist_ok=True)
    try: data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) or {"items": []}
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
    updates.append({"ts": datetime.datetime.utcnow().isoformat()+"Z","step": source_step,
                    "experiment_title": canonical_title,"added_alias": alias_title})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- step ----------

class LLMRefineTrialPlan(BaseStep):
    """
    Refine an executor-ready split plan. Supports generic row_filters and filename/dir globs.
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_refine_trial_plan",
            artifact="meta/trial_split_plans.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        art = project.artifacts_dir
        proposals = art / "meta" / "llm_propose_trial_candidates.json"
        probes    = art / "meta" / "candidate_probes.json"
        out       = art / self.artifact
        return proposals.exists() and probes.exists() and (self._force or not out.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        art = project.artifacts_dir
        proposals_all = _read_json(art / "meta" / "llm_propose_trial_candidates.json") or {}
        probes_all    = _read_json(art / "meta" / "candidate_probes.json") or {}

        cand_block = None
        if isinstance(proposals_all.get("items"), list) and idx < len(proposals_all["items"]):
            cand_block = proposals_all["items"][idx]
        probe_block = None
        if isinstance(probes_all.get("items"), list) and idx < len(probes_all["items"]):
            probe_block = probes_all["items"][idx]

        exp_title = item.get("title") or f"Experiment {idx+1}"
        exp_desc  = (item.get("standalone_summary") or "").strip()
        expected_n = _expected_subjects_from_filter_item(item)

        hint_text, hint_struct = _read_validation_hint_for_idx(project, idx)
        hint_lines = _format_hint_struct(hint_struct) if hint_struct else []

        # Record alias if proposals used a different title
        alt_from_proposals = (cand_block or {}).get("experiment_title")
        if alt_from_proposals and alt_from_proposals != exp_title:
            _append_alias(project, canonical_title=exp_title, alias_title=alt_from_proposals, source_step=self.name)

        cand_lines = _lines_from_candidates(cand_block or {})
        probe_lines = _lines_from_probes(probe_block or {})

        # ----- Strict schema: responses.parse requires 'required' to list ALL keys -----
        filter_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "column": {"type": "string"},
                "op": {"type": "string", "enum": ["==","!=", "in","not_in","contains","regex",">",">=","<","<="]},
                "value_str": {"type": "string"},
                "values": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["column","op","value_str","values"]
        }

        source_obj = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relpath": {"type": "string"},
                "sheet": {"type": "string"},
                "dir": {"type": "string"},
                "member_glob": {"type": "string"},
                "filename_id_regex": {"type": "string"},
                "subject_id_from": {"type": "string", "enum": ["filename","column","sheet"]},
                "subject_id_column": {"type": "string"},
                "experiment_column": {"type": "string"},
                "experiment_value": {"type": "string"},
                "row_filters": {"type": "array", "items": filter_item}
            },
            # IMPORTANT: include EVERY key in 'required' (OpenAI strict schema quirk)
            "required": [
                "relpath","sheet","dir","member_glob","filename_id_regex",
                "subject_id_from","subject_id_column","experiment_column","experiment_value","row_filters"
            ]
        }

        diagnostics_obj = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "notes": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "model": {"type": "string"}
            },
            "required": ["notes","confidence","model"]
        }

        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experiment_title": {"type": "string"},
                "strategy": {"type": "string", "enum": ["per_subject","per_experiment","one_file","none"]},
                "reason": {"type": "string"},
                "source": source_obj,
                "diagnostics": diagnostics_obj
            },
            "required": ["experiment_title","strategy","reason","source","diagnostics"]
        }

        # ----- Prompt -----
        parts: List[str] = []
        parts.append(f"TARGET EXPERIMENT (canonical): {exp_title}\n")
        if isinstance(expected_n, int) and expected_n > 0:
            parts.append(f"EXPECTED SUBJECTS (from paper/meta): ~{expected_n}\n")
        parts.append("DESCRIPTION:\n" + (exp_desc or "(none)") + "\n")
        parts.append("CANDIDATES (LLM pass 1):\n" + ("\n".join(cand_lines) if cand_lines else "(none)") + "\n")
        parts.append("PROBES (verified facts — includes ID-column uniqueness sketches):\n" + ("\n".join(probe_lines) if probe_lines else "(none)") + "\n")
        if hint_text: parts.append("VALIDATION FEEDBACK (previous attempt):\n" + hint_text + "\n")
        if hint_lines: parts.append("VALIDATION RECOMMENDATIONS (generic):\n" + "\n".join(hint_lines) + "\n")

        parts.append(
            "TASK:\n"
            "- Choose EXACTLY ONE strategy based on probes and validation feedback.\n"
            "- You MAY include row_filters to isolate the target subset (allowed ops: ==, !=, in, not_in, contains, regex, >, >=, <, <=).\n"
            "- If per_subject: set dir, member_glob, filename_id_regex (when parsing IDs), subject_id_from.\n"
            "- If per_experiment: set relpath(+sheet) and subject_id_column; row_filters optional.\n"
            "- If one_file: set relpath(+sheet), subject_id_column; row_filters optional.\n"
            "- Do NOT rename or drop columns; filtering is allowed.\n"
            "STRICT OUTPUT RULES:\n"
            "- Use only column names that appear in PROBES. If subject_id_from='column', subject_id_column MUST be one of subj_cols.\n"
            "- Any row_filters MUST use values observed in PROBES' uniqueness examples (do not invent unseen values).\n"
            "- For mixed files, prefer the discriminator column with smaller unique_count or clearer examples when choosing experiment filters.\n"
            "- Fill ALL fields in 'source'. For fields not applicable to the chosen strategy, set '' (empty) or [] (arrays like row_filters).\n"
            "- Only use relpaths/sheets that appear in CANDIDATES/PROBES. Keep experiment_title exactly as given."
        )
        prompt = "\n".join(parts)

        # ----- LLM call -----
        out = _generate(prompt, schema)
        if not isinstance(out, dict): out = {}

        # track alias if model rewrites title
        model_title = (out.get("experiment_title") or "").strip()
        if model_title and model_title != exp_title:
            _append_alias(project, canonical_title=exp_title, alias_title=model_title, source_step=self.name)

        # defaults & canonical title
        out.setdefault("strategy", "none")
        out.setdefault("reason", "")
        out.setdefault("source", {
            "relpath": "", "sheet": "", "dir": "", "member_glob": "", "filename_id_regex": "",
            "subject_id_from": "column", "subject_id_column": "", "experiment_column": "", "experiment_value": "",
            "row_filters": []
        })
        out.setdefault("diagnostics", {"notes": [], "confidence": 0.0, "model": ""})
        out["experiment_title"] = exp_title
        return out
