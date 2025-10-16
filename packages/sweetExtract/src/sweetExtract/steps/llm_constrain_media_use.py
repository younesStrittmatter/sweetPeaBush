# sweetExtract/steps/llm_constrain_media_use.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.catalog import Catalog
from sweetExtract.steps.asset_catalog import AssetCatalog
from sweetExtract.steps.llm_select_stimuli_per_role import LLMSelectStimuliPerRole

# ----------------------- tiny IO utils -----------------------

def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _save_meta(project: Project, filename: str, obj: Dict[str, Any]) -> Path:
    out = project.artifacts_dir / "meta" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

# ----------------------- upstream loaders -----------------------

def _load_selection(project: Project) -> List[Dict[str, Any]]:
    p = project.artifacts_dir / "meta" / "llm_stimuli_selection.json"
    obj = _read_json(p) or {}
    if isinstance(obj, dict) and obj.get("items"):  # multi-experiment
        return list(obj["items"])
    if isinstance(obj, dict) and obj.get("experiment_title"):  # single
        return [obj]
    return []

def _load_descriptions(project: Project) -> Dict[str, Dict[str, Any]]:
    p = project.artifacts_dir / "meta" / "describe_experiments.json"
    obj = _read_json(p) or {}
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        items = obj if isinstance(obj, list) else []
    by_title: Dict[str, Dict[str, Any]] = {}
    for it in items:
        t = (it or {}).get("title") or (it or {}).get("experiment_title") or ""
        if t:
            by_title[t] = it
    return by_title

def _load_asset_catalog(project: Project) -> Dict[str, Any]:
    return _read_json(project.artifacts_dir / "meta" / "asset_catalog.json") or {}

# ----------------------- LLM client -----------------------

def _generate_json(system_prompt: str, user_prompt: str,
                   schema: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=system_prompt,
        prompt=user_prompt,
        json_schema=schema,
        schema_name=schema_name,
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {}

# ----------------------- the step -----------------------

class LLMConstrainMediaUse(BaseStep):
    """
    A *light* pass to prevent overuse of media stimuli (Image/Video/Audio).
    It keeps media only when the role truly *requires* showing external media for construct validity.
    Otherwise it swaps to textual/symbolic stimuli (Text/HtmlText/Symbol/etc).

    No file-path resolution. No asset linking. This only decides the *final* stimulus type.

    Inputs:
      - meta/llm_stimuli_selection.json   (from LLMSelectStimuliPerRole)
      - meta/asset_catalog.json           (from AssetCatalog; for presence/absence context only)
      - meta/describe_experiments.json    (from DescribeExperiments; for task context)

    Output:
      - meta/llm_media_constrained.json
        {
          "items": [
            {
              "experiment_title": "...",
              "roles": [
                {
                  "role": "Sample cue (target exemplar)",
                  "original_stimulus": "Image",
                  "final_stimulus": "Image" | "Text" | "HtmlText" | "Symbol" | ...,
                  "media_required": true | false,
                  "justification": "brief rationale"
                },
                ...
              ],
              "asset_presence": { "images_count": N, "audio_count": M, "video_count": K }
            }
          ]
        }
    """
    artifact_is_list = True
    default_array_key = "items"

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_constrain_media_use",
            artifact="meta/llm_media_constrained.json",
            depends_on=[LLMSelectStimuliPerRole, AssetCatalog, Catalog, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        return True if self._force else not self.default_artifact(project).exists()

    # ------------ main (per experiment) ------------

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = (item.get("title") or item.get("experiment_title") or f"Experiment {idx+1}").strip()

        # Upstream artifacts
        selections = _load_selection(project)
        sel = next((s for s in selections if (s.get("experiment_title") or "").strip() == title), None)
        roles = list((sel or {}).get("roles") or [])

        descs = _load_descriptions(project)
        summary = (descs.get(title) or {}).get("standalone_summary") or (descs.get(title) or {}).get("summary") or ""

        asset_cat = _load_asset_catalog(project)
        images_count = int(((asset_cat.get("assets") or {}).get("images") or {}).get("count") or 0)
        audio_count  = int(((asset_cat.get("assets") or {}).get("audio")  or {}).get("count") or 0)
        video_count  = int(((asset_cat.get("assets") or {}).get("video")  or {}).get("count") or 0)

        # Compact role lines for prompt
        role_lines = [f"- {r.get('role','')} → {r.get('chosen','')}" for r in roles]

        # Allowed output stimuli: keep current + neutral textual/symbolic SweetBean classes
        allowed = sorted({
            *[r.get("chosen","") for r in roles if isinstance(r.get("chosen",""), str)],
            "Text","HtmlText","Symbol",
            "Fixation","Blank","HtmlKeyboardResponse","HtmlChoice",
            "Foraging","RSVP","Bandit","Feedback","Generic","Image","Video","Audio"
        })

        # ---------- LLM call ----------
        system = (
            "You are performing a *minimal* media necessity check for an experiment timeline.\n"
            "Goal: keep Image/Video/Audio only when the role *truly requires* showing external media to preserve the task's construct.\n"
            "If a symbolic or textual rendering conveys the same information (e.g., labels, simple icons, arrows), prefer Text/HtmlText/Symbol.\n"
            "If exemplar identity or rich pictorial content is essential (faces/objects where matching identity matters), keep Image.\n"
            "Remain neutral: do not downplay media needs; do not prefer text for convenience.\n"
            "Return JSON only."
        )

        user = (
            f"EXPERIMENT TITLE:\n{title}\n\n"
            "SUMMARY (truncated for context):\n"
            f"{summary[:4000]}\n\n"
            "ROLES (current choice):\n" + "\n".join(role_lines) + "\n\n"
            "ASSET PRESENCE (for context only; you are NOT resolving files):\n"
            f"- images_count: {images_count}\n- audio_count: {audio_count}\n- video_count: {video_count}\n\n"
            "ALLOWED_FINAL_STIMULI (must pick from this set):\n- " + "\n- ".join(allowed) + "\n\n"
            "For EACH role above, output an object with fields exactly:\n"
            "{ role, original_stimulus, final_stimulus, media_required, justification }\n"
            "Notes:\n"
            "• 'final_stimulus' must be in ALLOWED_FINAL_STIMULI.\n"
            "• Set media_required=true only if the role *cannot* fulfill its purpose with purely textual/symbolic presentation."
        )

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
                            "original_stimulus": {"type": "string"},
                            "final_stimulus": {"type": "string"},
                            "media_required": {"type": "boolean"},
                            "justification": {"type": "string"}
                        },
                        "required": ["role","original_stimulus","final_stimulus","media_required","justification"]
                    }
                }
            },
            "required": ["roles"]
        }

        out = _generate_json(system, user, schema, "LLMMediaConstrain") or {"roles": []}

        # ---------- sanitize & order-preserve ----------
        allowed_set = set(allowed)
        by_role = {r.get("role"): r for r in (out.get("roles") or [])}

        cleaned: List[Dict[str, Any]] = []
        for r in roles:  # preserve original order
            rn = r.get("role") or ""
            orig = r.get("chosen") or ""
            cand = by_role.get(rn, {})
            final = cand.get("final_stimulus") or orig
            if final not in allowed_set:
                final = orig if orig in allowed_set else "Text"
            media_required = bool(cand.get("media_required", False))
            just = cand.get("justification", "")
            cleaned.append({
                "role": rn,
                "original_stimulus": orig,
                "final_stimulus": final,
                "media_required": media_required,
                "justification": just
            })

        return {
            "experiment_title": title,
            "roles": cleaned,
            "asset_presence": {
                "images_count": images_count,
                "audio_count": audio_count,
                "video_count": video_count
            }
        }

    # map_over handles aggregation; compute() unused
    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError
