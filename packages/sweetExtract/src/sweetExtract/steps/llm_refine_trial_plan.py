# sweetExtract/steps/llm_refine_trial_plan.py
from __future__ import annotations
import os, json, datetime
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments

# ---------- small utils ----------

def _read_json(p: Path) -> Any:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _lines_from_candidates(cand_block: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for c in (cand_block.get("candidates") or [])[:4]:
        rp = c.get("relpath","")
        sh = c.get("sheet","")
        fmt = c.get("format","")
        sid = ", ".join((c.get("subject_id_hints") or [])[:2])
        tid = ", ".join((c.get("trial_index_hints") or [])[:2])
        lines.append(f"- relpath={rp} | sheet={sh} | format={fmt} | subject_hints=[{sid}] | trial_hints=[{tid}]")
    return lines

def _lines_from_probes(probe_block: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for p in (probe_block.get("probes") or [])[:6]:
        rp = p.get("relpath","")
        sh = p.get("sheet","")
        fmt = p.get("format","")
        cols = ", ".join((p.get("columns") or [])[:12])
        subj = ", ".join(p.get("present_subject_cols") or [])
        tri  = ", ".join(p.get("present_trial_cols") or [])
        exps = ", ".join(p.get("present_experiment_cols") or [])
        stat = p.get("stats") or {}
        er   = stat.get("error","")
        nr   = stat.get("n_rows", -1)
        nc   = stat.get("n_cols", 0)
        lines.append(f"- relpath={rp} | sheet={sh} | fmt={fmt} | n_rows~{nr} n_cols~{nc} err={er} | subj={subj} | trial={tri} | expm={exps} | cols=[{cols}]")
    return lines

def _generate(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {
            "experiment_title": "",
            "strategy": "none",
            "reason": "LLM unavailable",
            "source": {
                "relpath": "", "sheet": "",
                "dir": "", "member_glob": "", "filename_id_regex": "",
                "subject_id_from": "column", "subject_id_column": "",
                "experiment_column": "", "experiment_value": ""
            },
            "diagnostics": {"notes": ["LLM unavailable"], "confidence": 0.0, "model": ""}
        }

    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "Refine a plan to produce subject–trial-wise CSVs for ONE experiment.\n"
            "You have: experiment description, proposed candidates, and *verified probe facts* (headers, small previews).\n"
            "Pick EXACTLY ONE strategy:\n"
            "  1) per_subject — many per-subject files in ONE folder (headers similar). Provide: dir, member_glob, filename_id_regex; set subject_id_from.\n"
            "  2) per_experiment — ONE file per experiment; provide relpath(+sheet) and subject_id_column.\n"
            "  3) one_file — combined file; provide relpath(+sheet), subject_id_column, experiment_column, experiment_value.\n"
            "Do NOT drop/rename columns; we only split/copy. Prefer precise, minimal instructions grounded in the probes.\n"
            "Return ONLY JSON that matches the schema."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="TrialSplitPlanRefined",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return out or {}

# ---------- alias writer (inline, idempotent) ----------

def _append_alias(project: Project, canonical_title: str, alias_title: str, source_step: str) -> None:
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

# ---------- step ----------

class LLMRefineTrialPlan(BaseStep):
    """
    Use experiment description + LLM candidate proposals + probe facts to
    choose a concrete split strategy per experiment.

    Input:
      - meta/llm_propose_trial_candidates.json
      - meta/candidate_probes.json
      - meta/catalog_llm.json (optional context not attached, but snippets in prompt)

    Output:
      - meta/trial_split_plans.json (combined)
      - llm_refine_trial_plan/{idx}.json (per experiment)
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_refine_trial_plan",
            artifact="meta/trial_split_plans.json",
            depends_on=[DescribeExperiments],  # index-aligned mapping
            map_over=DescribeExperiments,
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

        # Record alias if proposals used a different title
        alt_from_proposals = (cand_block or {}).get("experiment_title")
        if alt_from_proposals and alt_from_proposals != exp_title:
            _append_alias(project, canonical_title=exp_title, alias_title=alt_from_proposals, source_step=self.name)

        cand_lines = _lines_from_candidates(cand_block or {})
        probe_lines = _lines_from_probes(probe_block or {})

        # Strict schema (executor-ready)
        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experiment_title": {"type": "string"},
                "strategy": {"type": "string", "enum": ["per_subject","per_experiment","one_file","none"]},
                "reason": {"type": "string"},
                "source": {
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
                        "experiment_value": {"type": "string"}
                    },
                    "required": [
                        "relpath","sheet","dir","member_glob","filename_id_regex",
                        "subject_id_from","subject_id_column","experiment_column","experiment_value"
                    ]
                },
                "diagnostics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "notes": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                        "model": {"type": "string"}
                    },
                    "required": ["notes","confidence","model"]
                }
            },
            "required": ["experiment_title","strategy","reason","source","diagnostics"]
        }

        prompt = (
            f"TARGET EXPERIMENT (canonical): {exp_title}\n\n"
            f"DESCRIPTION:\n{exp_desc}\n\n"
            "CANDIDATES (LLM pass 1):\n" + ("\n".join(cand_lines) if cand_lines else "(none)") + "\n\n"
            "PROBES (verified facts from small reads):\n" + ("\n".join(probe_lines) if probe_lines else "(none)") + "\n\n"
            "TASK:\n"
            "- Choose EXACTLY ONE strategy for this experiment based on probes and candidates.\n"
            "- If per_subject, set: dir (common folder), member_glob (e.g. '*.csv'), filename_id_regex, and subject_id_from (usually 'filename').\n"
            "- If per_experiment, set: relpath(+sheet) and subject_id_column.\n"
            "- If one_file, set: relpath(+sheet), subject_id_column, experiment_column, experiment_value.\n"
            "- Keep all columns; no renaming/dropping needed.\n"
            "- Return ONLY JSON matching the schema."
        )

        out = _generate(prompt, schema)

        # Fill safe defaults if degraded
        if not isinstance(out, dict):
            out = {}
        # If the model returned a different title, append it as an alias but KEEP canonical in artifact
        model_title = (out.get("experiment_title") or "").strip()
        if model_title and model_title != exp_title:
            _append_alias(project, canonical_title=exp_title, alias_title=model_title, source_step=self.name)

        out.setdefault("strategy", "none")
        out.setdefault("reason", "")
        out.setdefault("source", {
            "relpath": "", "sheet": "",
            "dir": "", "member_glob": "", "filename_id_regex": "",
            "subject_id_from": "column", "subject_id_column": "",
            "experiment_column": "", "experiment_value": ""
        })
        out.setdefault("diagnostics", {"notes": [], "confidence": 0.0, "model": ""})

        # enforce canonical title in the saved artifact
        out["experiment_title"] = exp_title
        return out
