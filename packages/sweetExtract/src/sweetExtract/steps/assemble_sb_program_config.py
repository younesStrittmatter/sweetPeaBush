from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_param_op_mappings import LLMParamOpMappings

def _read_json(p: Path)->Any:
    try:
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    except Exception: pass
    return None

def _write_json(p: Path, obj: Any)->None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _slug(s:str)->str:
    s=re.sub(r"\s+","-",str(s or "").strip()); s=re.sub(r"[^a-zA-Z0-9\-_.]","",s); return s.lower()[:120] or "exp"

class AssembleSBProgramConfig(BaseStep):
    """
    Inputs:
      - meta/llm_map_sb_parameters.json (refined)
      - meta/llm_param_op_mappings.json
      - meta/sb_trial_schema_for_llm.json
      - meta/llm_timeline_variables.json

    Output:
      - meta/sb_program_buildspec.json
        { "items":[
            {"experiment_title":"...", "slug":"...", "data_path":"...",
             "stimuli":[{"class_name":"Fixation","fixed_params":[{name,value_str},...],
                        "per_trial":[{"name":"...", "mapping": {...}}, ...]}],
             "timeline_variables":["colA","colB",...]}
          ]}
    """
    artifact_is_list=False
    default_array_key=None

    def __init__(self, force: bool=False):
        super().__init__(
            name="assemble_sb_program_config",
            artifact="meta/sb_program_buildspec.json",
            depends_on=[LLMParamOpMappings, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force=bool(force)

    def should_run(self, project: Project)->bool:
        out=project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    def _plan(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_map_sb_parameters.json") or {}
    def _ops(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_param_op_mappings.json") or {}
    def _schema(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json") or {}
    def _vars(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_timeline_variables.json") or {}

    def compute_one(self, project: Project, item: Dict[str,Any], idx:int,
                    all_items: List[Dict[str,Any]], prior: List[Dict[str,Any]])->Dict[str, Any]:
        title=(item or {}).get("title") or f"Experiment {idx+1}"
        slug=_slug(title)

        plan = self._plan(project)
        ops  = self._ops(project)
        ts   = self._schema(project)
        tv   = self._vars(project)

        pentry = next((e for e in (plan.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)), None) or {}
        stimuli=[]
        for u in (pentry.get("units") or []):
            stimuli.append({
                "class_name": u.get("stimulus") or "",
                "fixed_params": u.get("fixed_params") or [],
                "per_trial": [{"name": pt.get("name"), "mapping": {}} for pt in (u.get("per_trial") or []) if pt.get("name")]
            })

        mentry = next((e for e in (ops.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)), None) or {}
        mapping_list = mentry.get("mappings") or []
        # apply mappings by name
        mp = {m.get("param"): m.get("mapping") for m in mapping_list if m.get("param")}
        for s in stimuli:
            for pt in s["per_trial"]:
                name = pt.get("name")
                pt["mapping"] = mp.get(name, {"kind":"simple","source":"","ops":[],"fields":[],"items":[]})

        tsentry = next((e for e in (ts.get("items") or []) if (e.get("experiment_title")==title or e.get("title")==title)), None) or {}
        data_path = tsentry.get("data_path") or ""

        tventry = next((e for e in (tv.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)), None) or {}
        timeline_variables = tventry.get("timeline_variables") or []

        return {"experiment_title": title, "slug": slug, "data_path": data_path,
                "stimuli": stimuli, "timeline_variables": timeline_variables}

    def finalize(self, project: Project, results: List[Dict[str,Any]])->Dict[str,Any]:
        path = project.artifacts_dir / self.artifact
        _write_json(path, {"items": results})
        return {"items": results, "index": str(path.relative_to(project.artifacts_dir))}
