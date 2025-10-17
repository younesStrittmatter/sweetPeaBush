from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

# ---------- IO ----------
def _read_json(p: Path) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "-", str(s or "").strip())
    s = re.sub(r"[^a-zA-Z0-9\-_.]", "", s)
    return s.lower()[:120] or "exp"

# ---------- LLM ----------
def _generate_json(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {"timeline_variables": [], "param_to_columns": []}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "Select exactly which dataset columns must be exposed as SweetBean TimelineVariables. "
            "Use ONLY columns from the provided headers. "
            "Goal: minimal set that suffices to compute all per-trial parameters in the mapping plan. "
            "Do not invent columns. Output must match the schema exactly."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_SelectTimelineVars_V1",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {"timeline_variables": [], "param_to_columns": []}

class LLMSelectTimelineVariables(BaseStep):
    """
    Inputs:
      - meta/sb_trial_schema_for_llm.json
      - meta/llm_map_sb_parameters.json (refined; provides per_trial param names)
      - meta/experiments_empirical_detailed.json (context)

    Output:
      - meta/llm_timeline_variables.json
        { "items": [
            { "experiment_title": "...",
              "slug": "...",
              "timeline_variables": ["colA","colB",...],
              "param_to_columns": [ {"param":"<name>", "columns":["colX", ...]} ] }
        ]}
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_select_timeline_variables",
            artifact="meta/llm_timeline_variables.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force=bool(force)

    def should_run(self, project: Project) -> bool:
        out=project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # loaders
    def _trial_schema(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json"
        return _read_json(p) or {}

    def _mapping_plan(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "llm_map_sb_parameters_refined.json"
        return _read_json(p) or {}

    def _detailed(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "experiments_empirical_detailed.json"
        return _read_json(p) or {}

    def compute_one(self, project: Project, item: Dict[str, Any], idx:int,
                    all_items: List[Dict[str, Any]], prior: List[Dict[str, Any]])->Dict[str, Any]:
        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        ts = self._trial_schema(project)
        mp = self._mapping_plan(project)
        det = self._detailed(project)

        # trial entry
        tentry = next((e for e in (ts.get("items") or []) if (e.get("experiment_title")==title or e.get("title")==title)), None) or {}
        headers = tentry.get("headers") or []
        columns = tentry.get("columns") or {}
        data_path = tentry.get("data_path") or ""

        # mapping entry
        pentry = next((e for e in (mp.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)), None) or {}
        per_trial = []
        for u in (pentry.get("units") or []):
            for pt in (u.get("per_trial") or []):
                if pt.get("name"):
                    per_trial.append(pt.get("name"))

        # description text
        det_text = ""
        for e in (det.get("items") or []):
            nm = (e.get("title") or "").strip().lower()
            if nm == title.strip().lower() or (title.strip().lower() in [a.strip().lower() for a in (e.get("aliases") or [])]):
                det_text = (e.get("standalone_summary") or e.get("description") or e.get("methods_text") or "").strip()
                break

        # schema
        schema = {
            "type":"object","additionalProperties":False,
            "properties":{
                "timeline_variables":{"type":"array","items":{"type":"string"}},
                "param_to_columns":{"type":"array","items":{"$ref":"#/$defs/Map"}}
            },
            "required":["timeline_variables","param_to_columns"],
            "$defs":{
                "Map":{"type":"object","additionalProperties":False,
                       "properties":{"param":{"type":"string"},
                                     "columns":{"type":"array","items":{"type":"string"}}},
                       "required":["param","columns"]}
            }
        }

        prompt = (
            f"EXPERIMENT: {title}\n\n"
            f"COLUMNS+SAMPLES: {json.dumps([{ 'name':k, **(v or {}) } for k,v in columns.items()], ensure_ascii=False)[:4000]}\n\n"
            f"PER-TRIAL PARAM NAMES (from mapping plan): {per_trial}\n\n"
            f"DESCRIPTION: {det_text}\n\n"
            "Task: choose the MINIMAL set of dataset column names that must be bound as SweetBean TimelineVariables "
            "so that all listed per-trial parameters can be computed. Sometimes no columns are needed. Sometimes"
            "column names maybe ambiguous and it might be helpful to look at the sample values to disambiguate. "
            "Also return param→columns (each param may depend on one or more columns). "
            "Use ONLY provided header names. Do not invent columns."
        )

        out = _generate_json(prompt, schema)
        return {
            "experiment_title": title, "slug": slug,
            "timeline_variables": out.get("timeline_variables") or [],
            "param_to_columns": out.get("param_to_columns") or []
        }

    def finalize(self, project: Project, results: List[Dict[str, Any]])->Dict[str, Any]:
        path = project.artifacts_dir / self.artifact
        _write_json(path, {"items": results})
        return {"items": results, "index": str(path.relative_to(project.artifacts_dir))}
