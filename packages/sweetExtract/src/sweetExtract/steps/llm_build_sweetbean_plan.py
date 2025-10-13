# sweetExtract/steps/llm_build_sweetbean_plan.py
from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments

def _load_trial_map(art_dir, idx: int, title: str) -> Optional[Dict[str, Any]]:
    p = art_dir / "meta" / "llm_trial_column_map.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    items = (obj.get("items") or []) if isinstance(obj, dict) else []
    if 0 <= idx < len(items):
        return items[idx]
    for it in items:
        if (it or {}).get("experiment_title") == title:
            return it
    return None

def _alias_lookup(art_dir) -> Dict[str, int]:
    """
    Build alias -> trials_index idx map from exp_alias_index.json if present,
    otherwise fall back to canonical titles in trials_index.json.
    """
    out: Dict[str, int] = {}
    alias_p = art_dir / "meta" / "exp_alias_index.json"
    trials_p = art_dir / "meta" / "trials_index.json"

    if trials_p.exists():
        try:
            tobj = json.loads(trials_p.read_text(encoding="utf-8"))
            items = tobj.get("items") or []
            for i, it in enumerate(items):
                t = it.get("experiment_title")
                if isinstance(t, str) and t:
                    out.setdefault(t, i)
        except Exception:
            pass

    if alias_p.exists():
        try:
            aobj = json.loads(alias_p.read_text(encoding="utf-8"))
            idxmap = aobj.get("index") or {}
            for alias, rec in idxmap.items():
                i = rec.get("idx")
                if isinstance(i, int):
                    out[alias] = i
        except Exception:
            pass
    return out

def _pick_trials_item(art_dir, title: str) -> Optional[Dict[str, Any]]:
    alias = _alias_lookup(art_dir)
    trials_p = art_dir / "meta" / "trials_index.json"
    if not trials_p.exists():
        return None
    try:
        tobj = json.loads(trials_p.read_text(encoding="utf-8"))
        items = tobj.get("items") or []
    except Exception:
        return None
    # try alias index first
    if title in alias:
        i = alias[title]
        if 0 <= i < len(items):
            return items[i]
    # else try exact title
    for i, it in enumerate(items):
        if it.get("experiment_title") == title:
            return it
    return None

def _generate(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {
            "experiment_title": "",
            "sweetbean": {},
            "diagnostics": {"notes": ["LLM unavailable"], "confidence": 0.0, "model": ""}
        }

    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You convert a column/level mapping into a SweetBean plan.\n"
            "Define factors and levels, select a reasonable crossing, and specify dependent measures.\n"
            "Prefer the simplest design that captures the conditions; do not invent factors not suggested by the mapping.\n"
            "Return ONLY JSON matching the schema."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMSweetBeanPlan",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return out or {}

class LLMBuildSweetBeanPlan(BaseStep):
    """
    Build a SweetBean experiment plan via LLM using:
      - LLM trial column map (chosen columns + condition factors)
      - Experiment description (for naming & crossing hints)
      - trials_index/aliases (only to echo dest_dir/subjects context; no data read)

    Output: meta/llm_sweetbean_plan.json
            llm_sweetbean_plan/{idx}.json
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_build_sweetbean_plan",
            artifact="meta/llm_sweetbean_plan.json",
            depends_on=[DescribeExperiments],
            map_over=DescribeExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        have_map = (project.artifacts_dir / "meta" / "llm_trial_column_map.json").exists()
        have_trials = (project.artifacts_dir / "meta" / "trials_index.json").exists()
        dst = project.artifacts_dir / self.artifact
        if not (have_map and have_trials):
            return False
        if self._force:
            return True
        return not dst.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError  # mapped

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or f"Experiment {idx+1}"
        desc  = (item.get("standalone_summary") or "").strip()

        mapping = _load_trial_map(project.artifacts_dir, idx, title)
        if not mapping or mapping.get("status") not in {"ok", "warn"}:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_trial_column_mapping",
                "sweetbean": {},
                "diagnostics": {"notes": []}
            }

        t_item = _pick_trials_item(project.artifacts_dir, title) or {}
        dest_dir = t_item.get("dest_dir","")
        n_subjects = len(t_item.get("subjects") or [])

        # Compact mapping snippet for the LLM
        chosen = mapping.get("chosen_columns") or {}
        conds  = mapping.get("conditions") or []
        cond_lines = []
        for c in conds:
            nm = c.get("name","")
            src = ", ".join(c.get("source_columns") or [])
            lv  = ", ".join(c.get("levels") or [])
            vm  = "; ".join([f"{m.get('from','')}→{m.get('to','')}" for m in (c.get("value_map") or [])])
            cond_lines.append(f"- {nm} :: from [{src}] levels=[{lv}] map=[{vm}]")

        mapping_block = (
            "CHOSEN COLUMNS:\n"
            f"- subject={chosen.get('subject','')}\n"
            f"- trial={chosen.get('trial','')}\n"
            f"- rt={chosen.get('rt','')}\n"
            f"- accuracy={chosen.get('accuracy','')}\n"
            f"- response={chosen.get('response','')}\n"
            "CONDITION FACTORS:\n" + ("\n".join(cond_lines) if cond_lines else "(none)")
        )

        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experiment_title": {"type": "string"},
                "sweetbean": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "factors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "levels": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["name","levels"]
                            }
                        },
                        "crossing": {"type": "array", "items": {"type": "string"}},
                        "dependent_measures": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "column_bindings": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "subject": {"type": "string"},
                                "trial": {"type": "string"},
                                "rt": {"type": "string"},
                                "accuracy": {"type": "string"},
                                "response": {"type": "string"}
                            },
                            "required": ["subject","trial","rt","accuracy","response"]
                        },
                        "value_maps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "factor": {"type": "string"},
                                    "from": {"type": "string"},
                                    "to": {"type": "string"}
                                },
                                "required": ["factor","from","to"]
                            }
                        },
                        "notes": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["factors","crossing","dependent_measures","column_bindings","value_maps","notes"]
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
            "required": ["experiment_title","sweetbean","diagnostics"]
        }

        prompt = (
            f"TARGET EXPERIMENT: {title}\n"
            f"DESCRIPTION (standalone):\n{desc}\n\n"
            f"DATA CONTEXT: dest_dir={dest_dir} (subjects={n_subjects})\n\n"
            "TRIAL COLUMN/FACTOR MAPPING (from previous step):\n"
            f"{mapping_block}\n\n"
            "TASK:\n"
            "- Define SweetBean factors and levels from the condition mapping.\n"
            "- Choose a reasonable crossing; prefer a factorial crossing of the key factors.\n"
            "- Set dependent_measures (usually ['rt','accuracy']).\n"
            "- Provide column_bindings that point to the chosen data columns.\n"
            "- Include value_maps entries (factor/from→to) mirroring any cleanup needed for factor levels.\n"
            "- Keep it minimal and faithful to the mapping; do not invent factors not implied by the data."
        )

        out = _generate(prompt, schema)
        out.setdefault("experiment_title", title)

        # Echo back for convenience
        return {
            "experiment_title": out.get("experiment_title", title),
            "status": "ok",
            "sweetbean": out.get("sweetbean", {}),
            "diagnostics": out.get("diagnostics", {}),
            "context": {
                "dest_dir": dest_dir,
                "n_subjects": n_subjects,
            }
        }
