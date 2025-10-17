# sweetExtract/steps/llm_build_param_functions_positional.py
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_select_timeline_variables import LLMSelectTimelineVariables

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
        return {"functions": []}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You will output ONLY Python mapping functions for SweetBean parameters that must be computed per trial.\n"
            "Signature REQUIREMENT:\n"
            "  def map_<param>(x1, x2, ...):\n"
            "    # x1..xn are positional args corresponding to CSV columns, in the SAME ORDER as listed in 'args'\n"
            "Constraints:\n"
            "  • Use ONLY the provided CSV columns for each parameter (see 'tasks').\n"
            "  • Each function must be pure, deterministic, and use only Python builtins.\n"
            "  • Return values MUST match the parameter’s expected type/format (e.g., lists for RSVP streams; 0-based indices if a position, etc.).\n"
            "  • Parameter names and stimulus classes must match what is provided; do not invent.\n"
            "  • If no dynamic parameters exist, return an empty list.\n"
            "Return JSON that matches the provided schema exactly."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_BuildParamFunctions_Positional_V2",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {"functions": []}

class LLMBuildParamFunctionsPositional(BaseStep):
    """
    Build ONLY the per-trial mapping functions with positional arguments, one per parameter.

    Inputs (all must exist):
      - meta/experiments_empirical_detailed.json
      - meta/sb_trial_schema_for_llm.json
      - meta/sb_param_inventory_filtered_index.json
      - meta/llm_map_sb_parameters_refined.json   <-- used here
      - meta/llm_timeline_variables.json          <-- provides param→columns (positional)

    Output:
      - meta/llm_param_functions.json
        {
          "items": [
            {
              "experiment_title": "...",
              "slug": "...",
              "columns": [...],        # timeline variables
              "functions": [
                {
                  "stimulus": "RSVP",
                  "param": "streams",
                  "fn_name": "map_streams",
                  "args": [Timeline("Stream Left"),Timeline("Stream Right")],         # ORIGINAL CSV column names in order
                  "python_code": "def map_streams(stream_left, stream_right): ...",
                  "notes": ["..."]
                },
                ...
              ]
            }
          ]
        }
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_build_param_functions_positional",
            artifact="meta/llm_param_functions.json",
            depends_on=[LLMSelectTimelineVariables, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # ----- loaders -----
    def _detailed(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "experiments_empirical_detailed.json"
        if not p.exists(): raise FileNotFoundError("Missing: meta/experiments_empirical_detailed.json")
        return _read_json(p) or {}

    def _trial_schema(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json"
        if not p.exists(): raise FileNotFoundError("Missing: meta/sb_trial_schema_for_llm.json")
        obj = _read_json(p) or {}
        if not isinstance(obj.get("items"), list) or not obj["items"]:
            raise ValueError("sb_trial_schema_for_llm.json has no items[].")
        return obj

    def _filtered_idx(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "sb_param_inventory_filtered_index.json"
        if not p.exists(): raise FileNotFoundError("Missing: meta/sb_param_inventory_filtered_index.json")
        return _read_json(p) or {}

    def _refined_plan(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "llm_map_sb_parameters_refined.json"
        if p.exists(): return _read_json(p) or {}
        q = project.artifacts_dir / "meta" / "llm_map_sb_parameters.json"
        if q.exists(): return _read_json(q) or {}
        raise FileNotFoundError("Missing: meta/llm_map_sb_parameters_refined.json (or llm_map_sb_parameters.json)")

    def _timeline_vars(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "llm_timeline_variables.json"
        if not p.exists(): raise FileNotFoundError("Missing: meta/llm_timeline_variables.json")
        return _read_json(p) or {}

    # ----- helpers -----
    def _desc_for_title(self, detailed: Dict[str, Any], title: str) -> str:
        tnorm = (title or "").strip().lower()
        for it in (detailed.get("items") or []):
            t = (it.get("title") or "").strip().lower()
            aliases = [a.strip().lower() for a in (it.get("aliases") or [])]
            if t == tnorm or tnorm in aliases:
                return (it.get("standalone_summary") or it.get("description") or it.get("methods_text") or "").strip()
        return ""

    def _schema_entry(self, schema: Dict[str, Any], title: str) -> Dict[str, Any]:
        tnorm = (title or "").strip().lower()
        for it in (schema.get("items") or []):
            nm = (it.get("experiment_title") or it.get("title") or "").strip().lower()
            if nm == tnorm:
                return it
        return {}

    def _filtered_for_exp(self, inv: Dict[str, Any], title: str) -> Dict[str, Any]:
        tnorm = (title or "").strip().lower()
        for it in (inv.get("items") or []):
            nm = (it.get("experiment_title") or it.get("title") or it.get("slug") or "").strip().lower()
            if nm == tnorm:
                return it
        return {}

    def _timeline_for_exp(self, tlvars: Dict[str, Any], title: str) -> Dict[str, Any]:
        tnorm = (title or "").strip().lower()
        for it in (tlvars.get("items") or []):
            nm = (it.get("experiment_title") or it.get("title") or it.get("slug") or "").strip().lower()
            if nm == tnorm:
                return it
        return {}

    def compute_one(self, project: Project, item: Dict[str, Any], idx: int,
                    all_items: List[Dict[str, Any]], prior_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        detailed = self._detailed(project)
        schema   = self._trial_schema(project)
        inv      = self._filtered_idx(project)
        plan     = self._refined_plan(project)
        tlvars   = self._timeline_vars(project)

        desc_text  = self._desc_for_title(detailed, title)
        schema_ent = self._schema_entry(schema, title)
        inv_ent    = self._filtered_for_exp(inv, title)
        tl_ent     = self._timeline_for_exp(tlvars, title)

        # timeline variables chosen earlier + param->columns mapping (positional)
        columns_all = list(tl_ent.get("timeline_variables") or [])
        param_to_cols = {m.get("param"): list(m.get("columns") or []) for m in (tl_ent.get("param_to_columns") or [])}

        # Allowed param names per stimulus (for clamping)
        allowed_params_by_stim: Dict[str, List[str]] = {}
        for s in (inv_ent.get("items") or []):
            cls = s.get("class_name","")
            allowed_params_by_stim[cls] = [p.get("name") for p in (s.get("params") or []) if p.get("name")]

        # Per-trial param list comes FROM the refined plan (this experiment only)
        plan_ent = next(
            (e for e in (plan.get("items") or []) if (e.get("experiment_title")==title or e.get("slug")==slug)),
            {}
        )
        per_trial_params: List[tuple[str,str]] = []
        for unit in (plan_ent.get("units") or []):
            stim = unit.get("stimulus") or unit.get("class_name") or ""
            for pt in (unit.get("per_trial") or []):
                pname = pt.get("name") or ""
                if stim and pname:
                    per_trial_params.append((stim, pname))

        # Build function tasks: only those that (a) appear in plan, (b) have a columns list, (c) param is allowed for that stimulus
        tasks: List[Dict[str, Any]] = []
        for (stim, pname) in per_trial_params:
            cols = param_to_cols.get(pname, [])
            if not cols:
                continue
            if stim in allowed_params_by_stim and pname not in set(allowed_params_by_stim[stim]):
                # skip params not allowed for this stimulus per inventory clamp
                continue
            tasks.append({"stimulus": stim, "param": pname, "args": cols})

        # Minimal samples payload (to help LLM infer types safely)
        samples_payload: List[Dict[str, Any]] = []
        for name, meta in (schema_ent.get("columns") or {}).items():
            if name in columns_all:
                samples_payload.append({
                    "name": name,
                    "samples": (meta or {}).get("samples", [])[:6],
                    "n_unique": (meta or {}).get("n_unique", None)
                })

        # Allowed text for prompt
        stim_lines: List[str] = []
        for cls, names in allowed_params_by_stim.items():
            stim_lines.append(f"- {cls}: {', '.join(names)}")
        allowed_text = "\n".join(stim_lines) or "- (none)"

        # ------------- STRICT JSON SCHEMA -------------
        out_schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "functions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Func"}
                }
            },
            "required": ["functions"],
            "$defs": {
                "Func": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "stimulus": {"type": "string"},
                        "param": {"type": "string"},
                        "fn_name": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "string"}},   # original CSV headers in order
                        "python_code": {"type": "string"},                         # def map_<param>(arg1, arg2, ...):
                        "notes": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["stimulus","param","fn_name","args","python_code","notes"]
                }
            }
        }

        # ------------- PROMPT -------------
        prompt = (
            f"EXPERIMENT TITLE: {title}\n\n"
            "NATURAL-LANGUAGE DESCRIPTION:\n"
            f"{desc_text}\n\n"
            "ALLOWED STIMULI & PARAMETER NAMES:\n"
            f"{allowed_text}\n\n"
            "TIMELINE COLUMNS YOU MAY USE (names + example tokens):\n"
            f"{json.dumps(samples_payload, ensure_ascii=False)}\n\n"
            "TASKS (build one function with positional args matching 'args' order for each):\n"
            f"{json.dumps(tasks, ensure_ascii=False)}\n\n"
            "For each task, output an entry:\n"
            "  { stimulus, param, fn_name, args, python_code, notes }\n"
            "Rules:\n"
            "  • Function name must be 'map_<param>' in snake_case (param is exactly the parameter name).\n"
            "  • Positional arg names in the function signature should be valid identifiers derived from the CSV column names (snake_case),\n"
            "    but keep the ORIGINAL CSV header names in 'args' in the SAME ORDER for later binding.\n"
            "  • Use only Python builtins; implement type conversions, indexing shifts (e.g., 1→0), enum/boolean mapping, and list-building as needed.\n"
            "  • Return the correct type/shape for the parameter."
        )

        out = _generate_json(prompt, out_schema) or {"functions": []}

        return {
            "experiment_title": title,
            "slug": slug,
            "columns": columns_all,
            "functions": out.get("functions") or []
        }

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        path = project.artifacts_dir / self.artifact
        _write_json(path, {"items": results})
        return {"items": results, "index": str(path.relative_to(project.artifacts_dir))}
