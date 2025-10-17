from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_select_timeline_variables import LLMSelectTimelineVariables

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

def _gen(prompt:str, schema:Dict[str,Any])->Dict[str,Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {"mappings":[]}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL","gpt-5"),
        system_prompt=(
            "For each per-trial parameter, provide a deterministic mapping from dataset columns to the parameter value, "
            "using ONLY the allowed ops. Do not invent columns or params. Return JSON exactly as per schema."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_ParamOpMappings_V1",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {"mappings":[]}

class LLMParamOpMappings(BaseStep):
    """
    Inputs:
      - meta/llm_map_sb_parameters.json (refined; units with per_trial param names per stimulus)
      - meta/llm_timeline_variables.json (selected columns per experiment)
      - meta/sb_trial_schema_for_llm.json (headers + samples)
      - meta/experiments_empirical_detailed.json (context)

    Output:
      - meta/llm_param_op_mappings.json
        { "items":[
            {"experiment_title":"...", "slug":"...", "mappings":[
               {"param":"<name>", "mapping": { "kind":"simple|pack_object|pack_list",
                                               "source":"", "ops":[Op...], "fields":[...], "items":[...] }}
            ] }
          ]}
    """
    artifact_is_list=False
    default_array_key=None
    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_param_op_mappings",
            artifact="meta/llm_param_op_mappings.json",
            depends_on=[LLMSelectTimelineVariables, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force=bool(force)
    def should_run(self, project: Project)->bool:
        out=project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # loaders
    def _plan(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_map_sb_parameters_refined.json") or {}
    def _vars(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_timeline_variables.json") or {}
    def _schema(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json") or {}
    def _detailed(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "experiments_empirical_detailed.json") or {}

    def compute_one(self, project: Project, item: Dict[str,Any], idx:int,
                    all_items: List[Dict[str,Any]], prior: List[Dict[str,Any]])->Dict[str, Any]:
        title=(item or {}).get("title") or f"Experiment {idx+1}"
        slug=_slug(title)
        plan=self._plan(project); varsel=self._vars(project); ts=self._schema(project); det=self._detailed(project)

        pentry = next((e for e in (plan.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)), None) or {}
        per_trial_names=[]
        for u in (pentry.get("units") or []):
            for pt in (u.get("per_trial") or []):
                if pt.get("name"): per_trial_names.append(pt.get("name"))

        vsentry = next((e for e in (varsel.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)), None) or {}
        timeline_vars = vsentry.get("timeline_variables") or []
        param_to_cols = vsentry.get("param_to_columns") or []

        tsentry = next((e for e in (ts.get("items") or []) if (e.get("experiment_title")==title or e.get("title")==title)), None) or {}
        headers = tsentry.get("headers") or []
        columns = tsentry.get("columns") or {}
        data_path = tsentry.get("data_path") or ""

        det_text=""
        for e in (det.get("items") or []):
            nm=(e.get("title") or "").strip().lower()
            if nm==title.strip().lower() or (title.strip().lower() in [a.strip().lower() for a in (e.get("aliases") or [])]):
                det_text=(e.get("standalone_summary") or e.get("description") or e.get("methods_text") or "").strip()
                break

        # Strict schema for mappings
        schema = {
            "type":"object","additionalProperties":False,
            "properties":{"mappings":{"type":"array","items":{"$ref":"#/$defs/ParamMap"}}},
            "required":["mappings"],
            "$defs":{
                "Op":{"type":"object","additionalProperties":False,
                      "properties":{"op":{"type":"string"},"k":{"type":"string"},"min":{"type":"string"},"max":{"type":"string"},
                                    "pairs":{"type":"array","items":{"type":"object","additionalProperties":False,
                                                                     "properties":{"from":{"type":"string"},"to_str":{"type":"string"}},
                                                                     "required":["from","to_str"]}},
                                    "default_str":{"type":"string"},
                                    "allowed":{"type":"array","items":{"type":"string"}},
                                    "fallback":{"type":"string"},
                                    "value_str":{"type":"string"}},
                      "required":["op","k","min","max","pairs","default_str","allowed","fallback","value_str"]},
                "Mapping":{"type":"object","additionalProperties":False,
                           "properties":{"kind":{"type":"string"},
                                         "source":{"type":"string"},
                                         "ops":{"type":"array","items":{"$ref":"#/$defs/Op"}},
                                         "fields":{"type":"array","items":{"$ref":"#/$defs/Field"}},
                                         "items":{"type":"array","items":{"$ref":"#/$defs/Mapping"}}},
                           "required":["kind","source","ops","fields","items"]},
                "Field":{"type":"object","additionalProperties":False,
                         "properties":{"name":{"type":"string"},"mapping":{"$ref":"#/$defs/Mapping"}},
                         "required":["name","mapping"]},
                "ParamMap":{"type":"object","additionalProperties":False,
                            "properties":{"param":{"type":"string"},"mapping":{"$ref":"#/$defs/Mapping"}},
                            "required":["param","mapping"]}
            }
        }

        prompt = (
            f"EXPERIMENT: {title}\n"
            f"DATA PATH: {data_path}\n"
            f"HEADERS: {headers}\n"
            f"COLUMNS+SAMPLES: {json.dumps([{ 'name':k, **(v or {}) } for k,v in columns.items()], ensure_ascii=False)[:4000]}\n\n"
            f"DESCRIPTION: {det_text}\n\n"
            f"PER-TRIAL PARAMS: {per_trial_names}\n"
            f"TIMELINE VARIABLES (columns): {timeline_vars}\n"
            f"PARAM→COLUMNS (hints): {param_to_cols}\n\n"
            "Task: For EACH per-trial parameter, define a deterministic mapping using ONLY allowed ops:\n"
            "- to_string, to_lower, to_int, subtract, clamp, map_values, split_chars, ensure_in_set, constant.\n"
            "Use only provided column names. If a param is structured, use kind=pack_object or pack_list accordingly.\n"
            "Fill EVERY key required by the schema (unused → '' or [])."
        )

        out = _gen(prompt, schema) or {"mappings":[]}
        return {"experiment_title": title, "slug": slug, "mappings": out.get("mappings") or []}

    def finalize(self, project: Project, results: List[Dict[str, Any]])->Dict[str, Any]:
        path = project.artifacts_dir / self.artifact
        _write_json(path, {"items": results})
        return {"items": results, "index": str(path.relative_to(project.artifacts_dir))}
