from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional
from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

def _load_io_map(artifacts_dir, title: str) -> Dict[str, Any]:
    p = artifacts_dir / "meta" / "llm_experiment_io_map.json"
    if not p.exists(): return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    items = obj.get("items") or []
    for it in items:
        if (it or {}).get("experiment_title") == title and (it or {}).get("status") == "ok":
            return it
    return {}

def _generate_json(prompt: str) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}
    schema = {
        "type":"object","additionalProperties":False,
        "properties":{
            "paradigm":{"type":"string"},
            "timeline_roles":{"type":"array","items":{"type":"string"}},
            "notes":{"type":"array","items":{"type":"string"}}
        },
        "required":["paradigm","timeline_roles","notes"]
    }
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "Given an experiment summary and a data I/O sketch (column groups), "
            "name the likely paradigm and list a short sequence of generic roles "
            "e.g., Fixation, RSVP stream, Visual array, Response prompt, Feedback.\n"
            "Return JSON only."
        ),
        prompt=prompt, json_schema=schema, schema_name="LLMParadigm", strict_schema=True,
        reasoning_effort="low", text_verbosity="low"
    ) or {}

class LLMInferParadigm(BaseStep):
    """
    From a natural-language summary + I/O map, infer generic roles (no SB classes yet).
    Outputs: artifacts/llm_paradigm/{idx}.json
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_infer_paradigm",
            artifact="meta/llm_paradigm.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        return True if self._force else not self.default_artifact(project).exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or f"Experiment {idx+1}"
        summary = (item.get("standalone_summary") or "").strip()
        io = _load_io_map(project.artifacts_dir, title)
        pres = io.get("presentation") or []
        resp = io.get("response") or []
        pres_cols = sorted({c for g in pres for c in (g.get("columns") or [])})
        resp_cols = sorted({c for g in resp for c in (g.get("columns") or [])})

        prompt = (
            f"TITLE: {title}\n\nSUMMARY:\n{summary}\n\n"
            "PRESENTATION COLUMNS:\n- " + "\n- ".join(pres_cols) + "\n\n"
            "RESPONSE COLUMNS:\n- " + "\n- ".join(resp_cols) + "\n\n"
            "Return JSON with fields: paradigm, timeline_roles[], notes[]."
        )
        out = _generate_json(prompt) or {"paradigm":"", "timeline_roles":[], "notes":[]}
        out["experiment_title"] = title
        return out
