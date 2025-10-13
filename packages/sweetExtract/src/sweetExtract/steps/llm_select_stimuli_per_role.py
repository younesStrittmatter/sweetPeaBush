# sweetExtract/steps/llm_select_stimuli_per_role.py
from __future__ import annotations
import json, os
from importlib.resources import files
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments
from sweetExtract.steps.llm_infer_paradigm import LLMInferParadigm


# ----------------------- pkg data loaders -----------------------

def _pkg_json(pkg: str, name: str) -> Any:
    """Read a JSON resource bundled in the package (no envs, no artifacts)."""
    try:
        return json.loads(files(pkg).joinpath(name).read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_pkg_inventories() -> Dict[str, Any]:
    """
    Loads from sweetExtract/info/sb/{stimuli_index.json, init_docs.json, examples.json}.
    Supports list- or dict-shaped stimuli_index.
    """
    pkg = "sweetExtract.info.sb"
    return {
        "stimuli_index": _pkg_json(pkg, "stimuli_index.json"),
        "init_docs": _pkg_json(pkg, "init_docs.json"),
        "examples": _pkg_json(pkg, "examples.json"),
        "source_pkg": pkg,
    }

def _load_paradigm_map(project: Project) -> Dict[str, Dict[str, Any]]:
    """Return {title -> {'timeline_roles': [...], 'notes': [...]}} from meta/llm_paradigm.json."""
    p = project.artifacts_dir / "meta" / "llm_paradigm.json"
    try:
        obj = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        obj = {}
    if not obj:
        return {}
    items: List[Dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        items = obj["items"]
    elif isinstance(obj, dict) and obj.get("experiment_title"):
        items = [obj]
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        title = (it or {}).get("experiment_title") or ""
        if title:
            out[title] = dict(
                timeline_roles=list((it.get("timeline_roles") or [])),
                notes=list((it.get("notes") or [])),
            )
    return out


# ----------------------- LLM client -----------------------

def _generate_json(prompt: str) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "stimulus": {"type": "string"},
                                    "score": {"type": "number"},
                                    "why": {"type": "string"}
                                },
                                # OpenAI responses.parse wants all keys in 'required'
                                "required": ["stimulus", "score", "why"]
                            }
                        },
                        "chosen": {"type": "string"}
                    },
                    "required": ["role", "candidates", "chosen"]
                }
            },
            "gating": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "requirement": {"type": "string"},
                        "reason": {"type": "string"},
                        "roles": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["requirement", "reason", "roles"]
                }
            }
        },
        "required": ["roles", "gating"]
    }
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "Map generic timeline roles to concrete SweetBean stimulus classes.\n"
            "Use ONLY the provided inventory (names must match exactly). Prefer purpose-built classes "
            "(Fixation, Feedback, RSVP, HtmlChoice, HtmlKeyboardResponse, Foraging, etc.).\n"
            "Return JSON ONLY that matches the schema exactly."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMStimuliSelection",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {}


# ----------------------- the step -----------------------

class LLMSelectStimuliPerRole(BaseStep):
    """
    Per-experiment: map roles from LLMInferParadigm -> SweetBean stimuli.
    Writes meta/llm_stimuli_selection.json as items[].
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_select_stimuli_per_role",
            artifact="meta/llm_stimuli_selection.json",
            depends_on=[LLMInferParadigm, DescribeExperiments],
            map_over=DescribeExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        return True if self._force else not self.default_artifact(project).exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or f"Experiment {idx+1}"
        pmap = _load_paradigm_map(project)
        roles = list((pmap.get(title) or {}).get("timeline_roles") or [])

        inv = _load_pkg_inventories()
        stimuli_index = inv.get("stimuli_index")
        init_docs = inv.get("init_docs") or {}

        # ---- Build inventory from package data ----
        class_names: List[str] = []
        desc_by_name: Dict[str, str] = {}

        # stimuli_index: list or dict accepted
        if isinstance(stimuli_index, list):
            for it in stimuli_index:
                if isinstance(it, dict):
                    nm = it.get("title")
                    if nm:
                        class_names.append(nm)
                        if it.get("description"):
                            desc_by_name[nm] = it["description"]
        elif isinstance(stimuli_index, dict):
            if "stimuli" in stimuli_index and isinstance(stimuli_index["stimuli"], dict):
                for nm, meta in (stimuli_index["stimuli"] or {}).items():
                    class_names.append(nm)
                    if isinstance(meta, dict) and meta.get("description"):
                        desc_by_name[nm] = meta["description"]
            else:
                for nm, meta in stimuli_index.items():
                    class_names.append(nm)
                    if isinstance(meta, dict) and meta.get("description"):
                        desc_by_name[nm] = meta["description"]

        # augment with class_name from init_docs
        if isinstance(init_docs, dict):
            for _k, v in init_docs.items():
                if isinstance(v, dict):
                    nm = v.get("class_name")
                    if nm:
                        if nm not in class_names:
                            class_names.append(nm)
                        if nm not in desc_by_name:
                            ds = (v.get("summary") or v.get("init_docstring") or v.get("signature") or "")
                            if ds:
                                desc_by_name[nm] = str(ds)

        class_names = sorted({n for n in class_names if isinstance(n, str) and n.strip()})
        allowed = set(class_names)

        # No inventory? fallback (keeps pipeline alive)
        if not class_names:
            return {
                "experiment_title": title,
                "roles": [
                    {
                        "role": r,
                        "candidates": [{"stimulus": "Generic", "score": 0.5, "why": "no inventory"}],
                        "chosen": "Generic"
                    } for r in roles
                ],
                "gating": []
            }

        # Inventory lines for LLM
        inventory_lines: List[str] = []
        for nm in class_names:
            summary = (desc_by_name.get(nm) or "").strip()
            if len(summary) > 220: summary = summary[:217] + "..."
            inventory_lines.append(f"- {nm}: {summary}")

        prompt = (
            f"TITLE: {title}\n\n"
            "ROLES:\n- " + "\n- ".join(roles) + "\n\n"
            "INVENTORY (class: short purpose):\n" + "\n".join(inventory_lines) + "\n\n"
            "Return an object with keys 'roles' and 'gating' per the schema; "
            "gating may be an empty array, but MUST be present.\n"
            "Every candidate must include stimulus, score, and why."
        )
        out = _generate_json(prompt)

        # Sanitize to allowed class names
        cleaned_roles: List[Dict[str, Any]] = []
        for r in roles:
            m = next((x for x in (out.get("roles") or []) if (x.get("role") or "") == r), None)
            if not m:
                s0 = class_names[0]
                cleaned_roles.append({
                    "role": r,
                    "candidates": [{"stimulus": s0, "score": 0.5, "why": "fallback"}],
                    "chosen": s0
                })
                continue

            cand = []
            seen = set()
            for c in (m.get("candidates") or []):
                s = c.get("stimulus")
                if s in allowed and s not in seen:
                    seen.add(s)
                    cand.append({
                        "stimulus": s,
                        "score": float(c.get("score") or 0.5),
                        "why": c.get("why") or ""
                    })
            chosen = m.get("chosen")
            if chosen not in allowed:
                chosen = cand[0]["stimulus"] if cand else class_names[0]
            if not cand:
                cand = [{"stimulus": chosen, "score": 0.5, "why": "auto-selected"}]
            cleaned_roles.append({"role": r, "candidates": cand, "chosen": chosen})

        gating = out.get("gating") or []
        return {"experiment_title": title, "roles": cleaned_roles, "gating": gating}
