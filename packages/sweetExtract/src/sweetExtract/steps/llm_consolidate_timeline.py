# sweetExtract/steps/llm_consolidate_timeline.py
from __future__ import annotations
import json, os
from importlib.resources import files
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments
from sweetExtract.steps.llm_infer_paradigm import LLMInferParadigm
from sweetExtract.steps.llm_select_stimuli_per_role import LLMSelectStimuliPerRole

# ----------------------- pkg data loaders -----------------------

def _pkg_json(pkg: str, name: str) -> Any:
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

# ----------------------- upstream artifacts -----------------------

def _read_json(path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _load_paradigm_map(project: Project) -> Dict[str, Dict[str, Any]]:
    """Return {title -> {'timeline_roles': [...], 'notes': [...]}} from meta/llm_paradigm.json."""
    p = project.artifacts_dir / "meta" / "llm_paradigm.json"
    obj = _read_json(p) or {}
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

def _load_selection_sources(project: Project) -> Dict[str, Any]:
    """Load all potential selection sources."""
    meta = project.artifacts_dir / "meta"
    return {
        "media_constrained": _read_json(meta / "llm_media_constrained.json"),           # roles: [{role, original_stimulus, final_stimulus, ...}]
        "refined_selection": _read_json(meta / "llm_stimuli_selection_refined.json"),   # items[].roles[].chosen
        "selection":         _read_json(meta / "llm_stimuli_selection.json"),           # items[].roles[].chosen
    }

def _item_list(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("items"), list):
        return obj["items"]
    if obj.get("experiment_title") and obj.get("roles"):
        return [obj]
    return []

def _map_by_title(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        t = (it or {}).get("experiment_title") or (it or {}).get("title") or ""
        if t:
            out[t] = it
    return out

def _resolve_role_choices_for_title(sel_sources: Dict[str, Any], title: str) -> Dict[str, Any]:
    """
    Prefer media-constrained -> refined -> selection.
    Returns dict with keys: roles (list[{role, chosen}]), gating (list)
    """
    # 1) media_constrained
    mc_items = _item_list(sel_sources.get("media_constrained"))
    if mc_items:
        mc_map = _map_by_title(mc_items)
        mc = mc_map.get(title)
        if mc:
            roles = []
            for r in (mc.get("roles") or []):
                roles.append({
                    "role": r.get("role") or "",
                    "chosen": r.get("final_stimulus") or r.get("original_stimulus") or ""
                })
            # gating: pull from original selection if available
            gating = []
            sel_items = _item_list(sel_sources.get("selection"))
            sel_map = _map_by_title(sel_items)
            if sel_map.get(title):
                gating = sel_map[title].get("gating") or []
            return {"roles": roles, "gating": gating}

    # 2) refined_selection
    rs_items = _item_list(sel_sources.get("refined_selection"))
    if rs_items:
        rs_map = _map_by_title(rs_items)
        rs = rs_map.get(title)
        if rs:
            roles = [{"role": r.get("role") or "", "chosen": r.get("chosen") or ""} for r in (rs.get("roles") or [])]
            return {"roles": roles, "gating": rs.get("gating") or []}

    # 3) original selection
    sel_items = _item_list(sel_sources.get("selection"))
    if sel_items:
        sel_map = _map_by_title(sel_items)
        sl = sel_map.get(title)
        if sl:
            roles = [{"role": r.get("role") or "", "chosen": r.get("chosen") or ""} for r in (sl.get("roles") or [])]
            return {"roles": roles, "gating": sl.get("gating") or []}

    # default empty
    return {"roles": [], "gating": []}

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
            "timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "stimulus": {"type": "string"},
                        "covers_roles": {"type": "array", "items": {"type": "string"}},
                        "why": {"type": "string"}
                    },
                    "required": ["stimulus", "covers_roles", "why"]
                }
            },
            "drop_roles": {"type": "array", "items": {"type": "string"}},
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
        "required": ["timeline", "drop_roles", "gating"]
    }
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You transform a role→stimulus mapping into a MINIMAL external stimulus timeline.\n"
            "Some roles are INTERNAL to a stimulus and must NOT become separate timeline entries.\n"
            "Examples: within an RSVP, 'target frame', 'post-target frame', and 'remaining frames' are "
            "frames INTERNAL to the RSVP and should be covered by a SINGLE RSVP unit in the timeline.\n\n"
            "OUTPUT PRINCIPLES:\n"
            "1) Produce the smallest ordered list of SweetBean stimuli that can reproduce the experiment.\n"
            "2) For each timeline unit, list which input roles it COVERS.\n"
            "3) Any input role that is purely internal should appear in drop_roles (not in the external timeline).\n"
            "4) Only use stimulus class names from the provided inventory.\n"
            "Return JSON only that matches the schema exactly."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMConsolidatedTimeline",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low"
    ) or {}

# ----------------------- the step -----------------------

class LLMConsolidateTimeline(BaseStep):
    """
    Consolidate role→stimulus selections into a MINIMAL external timeline.

    Prefers post-media pass outputs when available:
      1) meta/llm_media_constrained.json (uses final_stimulus)
      2) meta/llm_stimuli_selection_refined.json (uses chosen)
      3) meta/llm_stimuli_selection.json (uses chosen)

    Inputs:
      - meta/llm_paradigm.json           (for original role order/context)
      - selection JSONs as above         (chosen/final_stimulus per role)
      - package inventories (sweetExtract/info/sb/*.json)

    Output:
      - meta/llm_stimuli_consolidation.json as items[]:
        {
          "experiment_title": "...",
          "timeline": [
            {"stimulus": "Fixation", "covers_roles": ["Fixation"], "why": "..."},
            {"stimulus": "RSVP", "covers_roles": ["Bilateral RSVP stream",
                                                  "Target frame (shape-defined digit)",
                                                  "Post-target distractor frame (same-stream digit or letters)",
                                                  "Remaining RSVP frames (letters)"], "why": "..."},
            {"stimulus": "Blank", "covers_roles": ["Stream offset"], "why": "..."},
            {"stimulus": "HtmlKeyboardResponse", "covers_roles": ["Response prompt (self-paced numeric key)"], "why": "..."},
            {"stimulus": "Blank", "covers_roles": ["ITI"], "why": "..."}
          ],
          "drop_roles": [],
          "gating": []
        }
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_consolidate_timeline",
            artifact="meta/llm_stimuli_consolidation.json",
            depends_on=[LLMSelectStimuliPerRole, LLMInferParadigm, DescribeExperiments],
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

        # Load role order + notes (for context)
        pmap = _load_paradigm_map(project)
        roles_in_order = list((pmap.get(title) or {}).get("timeline_roles") or [])
        roles_set = set(roles_in_order)

        # Load per-role choices (preferring media-constrained/refined selection)
        sel_sources = _load_selection_sources(project)
        resolved = _resolve_role_choices_for_title(sel_sources, title)
        chosen_roles: List[Dict[str, Any]] = resolved["roles"]
        gating_from_sel = resolved.get("gating") or []

        # Build a compact list for the prompt showing role -> chosen stimulus
        role_lines: List[str] = []
        chosen_names: List[str] = []
        chosen_by_role: Dict[str, str] = {}
        for r in chosen_roles:
            rn = r.get("role") or ""
            ch = r.get("chosen") or ""
            if rn:
                role_lines.append(f"- {rn} -> {ch}")
                chosen_by_role[rn] = ch
            if ch:
                chosen_names.append(ch)

        # Inventory from package (for name clamping)
        inv = _load_pkg_inventories()
        stimuli_index = inv.get("stimuli_index")
        init_docs = inv.get("init_docs") or {}

        allowed: List[str] = []
        if isinstance(stimuli_index, list):
            allowed = [it.get("title") for it in stimuli_index if isinstance(it, dict) and it.get("title")]
        elif isinstance(stimuli_index, dict):
            if "stimuli" in stimuli_index and isinstance(stimuli_index["stimuli"], dict):
                allowed = list((stimuli_index["stimuli"] or {}).keys())
            else:
                allowed = list(stimuli_index.keys())

        if isinstance(init_docs, dict):
            for v in init_docs.values():
                if isinstance(v, dict) and v.get("class_name"):
                    if v["class_name"] not in allowed:
                        allowed.append(v["class_name"])

        # Fall back to whatever was chosen, if package inventory is not available
        if not allowed:
            allowed = sorted({a for a in chosen_names if a})
        allowed = sorted({a for a in allowed if a})
        allowed_set = set(allowed)

        # LLM prompt
        prompt = (
            f"TITLE: {title}\n\n"
            "INPUT ROLES IN ORDER:\n- " + "\n- ".join(roles_in_order) + "\n\n"
            "ROLE → CHOSEN STIMULUS (post-media pass if available):\n" + "\n".join(role_lines) + "\n\n"
            "INVENTORY (allowed class names only):\n- " + "\n- ".join(allowed) + "\n\n"
            "Task: produce a MINIMAL external stimulus timeline that can reproduce the experiment.\n"
            "- Merge roles that are internal to the same stimulus into a single timeline unit.\n"
            "- Preserve the original ordering implied by INPUT ROLES.\n"
            "- Output 'timeline' (ordered units), 'drop_roles' (roles internal to other stimuli), and 'gating' (may be []).\n"
            "Return JSON ONLY matching the schema."
        )

        out = _generate_json(prompt) or {"timeline": [], "drop_roles": [], "gating": []}

        # Sanitize: clamp names to allowed, ensure covers_roles are subset of input roles
        cleaned_tl: List[Dict[str, Any]] = []
        for u in (out.get("timeline") or []):
            stim = u.get("stimulus") or ""
            cov = [r for r in (u.get("covers_roles") or []) if r in roles_set]
            if stim in allowed_set and cov:
                cleaned_tl.append({"stimulus": stim, "covers_roles": cov, "why": u.get("why") or ""})

        drop_roles = [r for r in (out.get("drop_roles") or []) if r in roles_set]
        # Remove any drop_roles that are still covered in the timeline (timeline wins)
        covered = {r for u in cleaned_tl for r in (u.get("covers_roles") or [])}
        drop_roles = [r for r in drop_roles if r not in covered]

        # If LLM failed to consolidate anything, produce a trivial minimal grouping (fallback only)
        if not cleaned_tl:
            last_stim, bucket = None, []
            order = []
            for rn in roles_in_order:
                st = chosen_by_role.get(rn, "")
                if st != last_stim and bucket:
                    order.append({"stimulus": last_stim, "covers_roles": bucket, "why": "fallback grouping"})
                    bucket = []
                last_stim = st
                bucket.append(rn)
            if bucket:
                order.append({"stimulus": last_stim, "covers_roles": bucket, "why": "fallback grouping"})
            cleaned_tl = [u for u in order if (u.get("stimulus") in allowed_set and u.get("covers_roles"))]

        # Merge any gating the LLM suggested with gating from selection (dedupe lightly)
        gating_llm = out.get("gating") or []
        def _gkey(g): return (g.get("requirement",""), g.get("reason",""), tuple(g.get("roles") or []))
        seen = {_gkey(g) for g in gating_from_sel}
        merged_gating = list(gating_from_sel)
        for g in gating_llm:
            k = _gkey(g)
            if k not in seen:
                merged_gating.append(g); seen.add(k)

        return {
            "experiment_title": title,
            "timeline": cleaned_tl,
            "drop_roles": drop_roles,
            "gating": merged_gating
        }
