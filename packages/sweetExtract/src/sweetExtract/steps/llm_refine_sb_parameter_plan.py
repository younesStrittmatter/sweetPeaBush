from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_map_sb_parameters import LLMMapSBParameters

# ---------------- IO ----------------
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

# --------------- LLM ---------------
def _generate_json(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You are given a preliminary SweetBean parameter plan (possibly imperfect). "
            "Refine it so it is runnable and consistent with: "
            "(1) allowed parameters per stimulus class, "
            "(2) REQUIRED trial schema (column names + samples), and "
            "(3) the natural-language experiment description. "
            "Do not invent columns; only map from provided headers. "
            "Do not invent parameter names; only use allowed names per class. "
            "If a parameter is not needed, omit it or leave it to defaults. "
            "IMPORTANT: Output MUST match the JSON schema exactly. "
            "Every required key must be present; if unused, set to '' (empty string) or [] (empty array)."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_SBParamPlan_Refined_Generic_V1",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {}

class LLMRefineSBParameterPlan(BaseStep):
    """
    LLM-only, stimulus-agnostic refinement of llm_map_sb_parameters.

    Inputs (must exist):
      - artifacts/meta/llm_map_sb_parameters.json         (from LLMMapSBParameters)
      - artifacts/meta/sb_param_inventory_filtered_index.json
      - artifacts/meta/sb_trial_schema_for_llm.json
      - artifacts/meta/experiments_empirical_detailed.json

    Output (overwrites the same file; no new files):
      - artifacts/meta/llm_map_sb_parameters.json
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_refine_sb_parameter_plan",
            artifact="meta/llm_map_sb_parameters_refined.json",
            depends_on=[LLMMapSBParameters, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        # Always run so we can refine in place
        return True

    # ----- loaders -----
    def _plan_index(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "llm_map_sb_parameters.json"
        if not p.exists():
            raise FileNotFoundError("Missing required: artifacts/meta/llm_map_sb_parameters.json")
        return _read_json(p) or {}

    def _filtered_index(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "sb_param_inventory_filtered_index.json"
        if not p.exists():
            raise FileNotFoundError("Missing required: artifacts/meta/sb_param_inventory_filtered_index.json")
        return _read_json(p) or {}

    def _trial_schema(self, project: Project) -> Dict[str, Any]:
        p = project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json"
        if not p.exists():
            raise FileNotFoundError("Missing required: artifacts/meta/sb_trial_schema_for_llm.json")
        obj = _read_json(p) or {}
        if not isinstance(obj.get("items"), list) or not obj["items"]:
            raise ValueError("sb_trial_schema_for_llm.json has no items[].")
        return obj

    def _detailed_description(self, project: Project, title: str) -> str:
        p = project.artifacts_dir / "meta" / "experiments_empirical_detailed.json"
        if not p.exists():
            return ""
        obj = _read_json(p) or {}
        items = obj.get("items") or []
        tnorm = title.strip().lower()
        for it in items:
            t = (it.get("title") or "").strip().lower()
            aliases = [a.strip().lower() for a in (it.get("aliases") or [])]
            if t == tnorm or tnorm in aliases:
                return (it.get("standalone_summary") or it.get("description") or it.get("methods_text") or "").strip()
        return ""

    def _allowed_params_text(self, inv_index: Dict[str, Any], title: str) -> str:
        """Compile allowed param list per class for the specific experiment."""
        items = inv_index.get("items") or []
        slug = _slug(title)
        entry = next((x for x in items if x.get("experiment_title")==title or x.get("slug")==slug), None)
        if not entry:
            return "- (none)"
        lines = []
        for it in entry.get("items", []):
            cls = it.get("class_name")
            names = [p.get("name") for p in (it.get("params") or []) if p.get("name")]
            if cls:
                lines.append(f"- {cls}: {', '.join(names)}")
        return "\n".join(lines) if lines else "- (none)"

    def compute_one(self, project: Project, item: Dict[str, Any], idx: int,
                    all_items: List[Dict[str, Any]], prior: List[Dict[str, Any]]) -> Dict[str, Any]:

        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        # load inputs
        plan_idx = self._plan_index(project)
        inv_idx  = self._filtered_index(project)
        trial_s  = self._trial_schema(project)

        # current plan entry for this experiment
        existing_items = plan_idx.get("items") or []
        curr = next((x for x in existing_items if (x.get("experiment_title")==title or x.get("slug")==slug)), None)
        if not curr:
            raise ValueError(f"No plan entry found in llm_map_sb_parameters.json for '{title}'.")

        # trial schema entry (columns + samples)
        titems = trial_s.get("items") or []
        schema_ent = next((ent for ent in titems if ent.get("experiment_title")==title or ent.get("title")==title), None)
        if not schema_ent:
            raise ValueError(f"No trial schema entry found for '{title}' in sb_trial_schema_for_llm.json.")
        headers = schema_ent.get("headers") or []
        columns = schema_ent.get("columns") or {}
        cols_for_prompt = []
        for name, meta in columns.items():
            cols_for_prompt.append({"name": name, "samples": (meta or {}).get("samples", [])[:6], "n_unique": (meta or {}).get("n_unique", None)})

        # allowed params per class
        inv_text = self._allowed_params_text(inv_idx, title)

        # description text
        desc_text = self._detailed_description(project, title) or ""

        # -------- STRICT SCHEMA (top-level $defs only; no unions; all keys present) --------
        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "units": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Unit"}
                }
            },
            "required": ["units"],
            "$defs": {
                "FixedParam": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "value_str": {"type": "string"}},
                    "required": ["name","value_str"]
                },
                "FixedProv": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "provenance": {"type": "string"}},
                    "required": ["name","provenance"]
                },
                "Op": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "op": {"type": "string"},
                        "k": {"type": "string"},
                        "min": {"type": "string"},
                        "max": {"type": "string"},
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {"from": {"type": "string"}, "to_str": {"type": "string"}},
                                "required": ["from","to_str"]
                            }
                        },
                        "default_str": {"type": "string"},
                        "allowed": {"type": "array", "items": {"type": "string"}},
                        "fallback": {"type": "string"},
                        "value_str": {"type": "string"}
                    },
                    "required": ["op","k","min","max","pairs","default_str","allowed","fallback","value_str"]
                },
                "Mapping": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string"},    # 'simple'|'pack_object'|'pack_list'
                        "source": {"type": "string"},  # for 'simple', else ''
                        "ops": {"type": "array", "items": {"$ref": "#/$defs/Op"}},
                        "fields": {"type": "array", "items": {"$ref": "#/$defs/MappingField"}},
                        "items": {"type": "array", "items": {"$ref": "#/$defs/Mapping"}}
                    },
                    "required": ["kind","source","ops","fields","items"]
                },
                "MappingField": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "mapping": {"$ref": "#/$defs/Mapping"}},
                    "required": ["name","mapping"]
                },
                "PerTrial": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "mapping": {"$ref": "#/$defs/Mapping"}},
                    "required": ["name","mapping"]
                },
                "Unit": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "stimulus": {"type": "string"},
                        "fixed_params": {"type": "array", "items": {"$ref": "#/$defs/FixedParam"}},
                        "fixed_provenance": {"type": "array", "items": {"$ref": "#/$defs/FixedProv"}},
                        "per_trial": {"type": "array", "items": {"$ref": "#/$defs/PerTrial"}},
                        "leave_unset_to_default": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["stimulus","fixed_params","fixed_provenance","per_trial","leave_unset_to_default","notes"]
                }
            }
        }

        # -------- General, stimulus-agnostic normalization catalog (for the LLM) --------
        normalization_catalog = [
            # Lists vs scalars
            "If a parameter expects a list of tokens (e.g., a sequence), convert string columns to lists using an op like 'split_chars'.",
            # Indices / positions
            "If a parameter is an index/position, ensure 0-based indexing. When the dataset is 1-based, add ops: to_int, subtract{k: '1'}.",
            # Enumerations / aliases
            "Map enumerated values via 'map_values' (e.g., side codes, condition labels, shapes), and set a sensible default_str.",
            # Booleans
            "Normalize boolean-like data (0/1, yes/no, true/false) via 'map_values' to 'true'/'false' strings when used in fixed_params.",
            # CSS/units
            "For visual stroke/size parameters that are CSS-like, ensure units (e.g., '4px').",
            # Clamping
            "If an index might overflow the available items, include a 'clamp' op with min/max derived from context when appropriate.",
            # Defaults
            "Parameters that are not required should be omitted or placed in leave_unset_to_default so runtime defaults apply.",
            # Structured params
            "For complex objects or arrays, use 'pack_object' and 'pack_list' with nested simple mappings.",
        ]

        # ---------------- prompt ----------------
        prompt = (
            f"EXPERIMENT: {title}\n\n"
            "CURRENT PLAN (to be refined):\n"
            f"{json.dumps({'units': curr.get('units', [])}, ensure_ascii=False)}\n\n"
            "ALLOWED PARAMS PER STIMULUS:\n"
            f"{inv_text}\n\n"
            "TRIAL SCHEMA (columns + samples):\n"
            f"{json.dumps(cols_for_prompt, ensure_ascii=False)}\n\n"
            "ALL HEADERS:\n"
            f"{schema_ent.get('headers', [])}\n\n"
            "DESCRIPTION TEXT (from experiments_empirical_detailed.json):\n"
            f"{desc_text}\n\n"
            "GENERAL NORMALIZATION CATALOG:\n- " + "\n- ".join(normalization_catalog) + "\n\n"
            "Refinement Requirements:\n"
            "- Make the plan runnable and deterministic without introducing new dataset columns.\n"
            "- Only use parameter names listed as allowed for each stimulus class; drop or rename others.\n"
            "- If a parameter needs transformation (e.g., type conversion, indexing, string splitting, enum mapping), "
            "  add the appropriate ops chain using ONLY the allowed ops: to_string, to_lower, to_int, subtract, clamp, "
            "  map_values, split_chars, ensure_in_set, constant.\n"
            "- If a required key in the schema is not used, set it to '' (string) or [] (array)."
        )

        out = _generate_json(prompt, schema) or {"units": []}
        return {"experiment_title": title, "slug": slug, **out}

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Overwrite the same artifact the previous step produced.
        path = project.artifacts_dir / self.artifact
        _write_json(path, {"items": results})
        return {"items": results, "index": str(path.relative_to(project.artifacts_dir))}
