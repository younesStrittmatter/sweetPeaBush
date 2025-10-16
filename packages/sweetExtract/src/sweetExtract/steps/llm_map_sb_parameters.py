from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

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
            "Map SweetBean stimulus parameters from: "
            "(1) allowed params per class, (2) REQUIRED trial schema (columns+samples), "
            "(3) experiment description. "
            "ONLY use allowed parameter names. Leave unused params to defaults. "
            "IMPORTANT: You MUST fill every key required by the JSON schema. "
            "If a field is not used, set it to '' (empty string) for scalars or [] for arrays. "
            "Return ONLY JSON that matches the schema exactly."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_SBParamMap_UsingTrialSchema_V6",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {}

class LLMMapSBParameters(BaseStep):
    """
    REQUIRED inputs:
      - artifacts/meta/sb_param_inventory_filtered_index.json
      - artifacts/meta/sb_trial_schema_for_llm.json
      - artifacts/meta/experiments_empirical_detailed.json

    Output (single file):
      - artifacts/meta/llm_map_sb_parameters.json  with:
        {
          "items": [
            {
              "experiment_title": "...",
              "slug": "...",
              "units": [ ... ]   # per the schema
            },
            ...
          ]
        }
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_map_sb_parameters",
            artifact="meta/llm_map_sb_parameters.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # ----- loaders -----
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

    def compute_one(self, project: Project, item: Dict[str, Any], idx: int,
                    all_items: List[Dict[str, Any]], prior: List[Dict[str, Any]]) -> Dict[str, Any]:
        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        # allowed params per class
        fidx_items = (self._filtered_index(project).get("items") or [])
        per_exp_inv = next((ent for ent in fidx_items
                            if ent.get("slug")==slug or ent.get("experiment_title")==title), None)
        if not per_exp_inv or not per_exp_inv.get("items"):
            raise ValueError(f"No filtered inventory found for '{title}'.")
        inv_map = { it.get("class_name"): it for it in per_exp_inv["items"] }

        # trial schema (columns + samples)
        titems = self._trial_schema(project).get("items") or []
        schema_ent = next((ent for ent in titems
                           if ent.get("experiment_title")==title or ent.get("title")==title), None)
        if not schema_ent:
            raise ValueError(f"No trial schema entry found for '{title}'.")
        headers = schema_ent.get("headers") or []
        columns = schema_ent.get("columns") or {}

        # allowed param text for prompt
        inv_lines: List[str] = []
        for cls, ent in inv_map.items():
            param_names = [p.get("name") for p in (ent.get("params") or []) if p.get("name")]
            inv_lines.append(f"- {cls}: {', '.join(param_names)}")
        inv_text = "\n".join(inv_lines) or "- (none)"

        # compact columns preview for prompt
        cols_for_prompt: List[Dict[str, Any]] = []
        for name, meta in columns.items():
            cols_for_prompt.append({
                "name": name,
                "samples": (meta or {}).get("samples", [])[:6],
                "n_unique": (meta or {}).get("n_unique", None)
            })

        # -------- STRICT SCHEMA (top-level $defs only) --------
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
                    "properties": {
                        "name": {"type": "string"},
                        "value_str": {"type": "string"}
                    },
                    "required": ["name","value_str"]
                },
                "FixedProv": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "provenance": {"type": "string"}
                    },
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
                                "properties": {
                                    "from": {"type": "string"},
                                    "to_str": {"type": "string"}
                                },
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
                        "kind": {"type": "string"},
                        "source": {"type": "string"},
                        "ops": {"type": "array", "items": {"$ref": "#/$defs/Op"}},
                        "fields": {"type": "array", "items": {"$ref": "#/$defs/MappingField"}},
                        "items": {"type": "array", "items": {"$ref": "#/$defs/Mapping"}}
                    },
                    "required": ["kind","source","ops","fields","items"]
                },
                "MappingField": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "mapping": {"$ref": "#/$defs/Mapping"}
                    },
                    "required": ["name","mapping"]
                },
                "PerTrial": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "mapping": {"$ref": "#/$defs/Mapping"}
                    },
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

        # ------------- prompt -------------
        desc_text = self._detailed_description(project, title) or ""
        prompt = (
            f"EXPERIMENT: {title}\n\n"
            "ALLOWED PARAMS PER STIMULUS:\n"
            f"{inv_text}\n\n"
            "TRIAL SCHEMA (columns + samples):\n"
            f"{json.dumps(cols_for_prompt, ensure_ascii=False)}\n\n"
            "ALL HEADERS:\n"
            f"{headers}\n\n"
            "DESCRIPTION TEXT (from experiments_empirical_detailed.json):\n"
            f"{desc_text}\n\n"
            "Instructions:\n"
            "- Output MUST satisfy the schema. If a field doesn't apply, use '' (empty string) or [] (empty array).\n"
            "- fixed_params: list of {name, value_str} constants (numbers as strings, e.g., '50').\n"
            "- fixed_provenance: list of {name, provenance} with provenance in ['paper','llm_inferred','dataset']\n"
            "- per_trial: list of {name, mapping}. mapping:\n"
            "    kind: 'simple'|'pack_object'|'pack_list';\n"
            "    source: column name for 'simple', else '';\n"
            "    ops: array of ops (allowed op names); unused => [];\n"
            "    fields: for 'pack_object' (array of {name, mapping}); else [];\n"
            "    items: for 'pack_list' (array of mapping); else [].\n"
            "- For each op object, ALWAYS include keys: op, k, min, max, pairs, default_str, allowed, fallback, value_str (unused => '' or []).\n"
            "- Use ONLY allowed parameter names for each stimulus."
        )

        out = _generate_json(prompt, schema) or {"units": []}
        # return in-memory; DO NOT write per-experiment files
        return {"experiment_title": title, "slug": slug, **out}

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # single artifact only
        idx_path = project.artifacts_dir / self.artifact
        _write_json(idx_path, {"items": results})
        return {"items": results, "index": str(idx_path.relative_to(project.artifacts_dir))}
