# sweetExtract/steps/llm_blocks_from_description.py
from __future__ import annotations
import json, os, textwrap
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

# ---------- helpers ----------

def _read_json(path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_descriptions(project: Project) -> Dict[str, Dict[str, Any]]:
    """
    Load the detailed experiment descriptions.
    Expected: artifacts/meta/experiments_detailed.json from DescribeExperiments:
      { "items": [ { "title": ..., "standalone_summary": ..., "aliases": [...] }, ... ] }
    """
    p = project.artifacts_dir / "meta" / "experiments_detailed.json"
    obj = _read_json(p) or {}
    items = obj.get("items") if isinstance(obj, dict) else None
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for it in items:
            title = (it or {}).get("title") or (it or {}).get("experiment_title") or ""
            if title:
                out[title] = it
    return out

# ---------- LLM call ----------

def _generate_from_description(prompt: str) -> Dict[str, Any]:
    """
    Call the LLM with a strict JSON schema. All properties are listed in `required`
    (nullable where appropriate) so parse() won’t reject the response.
    """
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}

    # Per-block conceptual description (no data columns here)
    block_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},                 # e.g., "Practice", "Speed", "Accuracy", "Block 1"
            "category": {"type": "string"},              # "practice" | "training" | "main" | "phase" | "condition" | "test" | "other"
            "repeats": {"type": ["integer","null"]},     # per subject, if stated
            "n_trials_expected": {"type": ["integer","null"]},
            "notes": {"type": ["string","null"]},        # brief
            "evidence": { "type": "array", "items": {"type":"string"} },  # short phrases supporting this block
            "confidence": {"type": "number"}             # 0..1
        },
        "required": ["label","category","repeats","n_trials_expected","notes","evidence","confidence"]
    }

    # Whole plan from prose only
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "block_count": {"type": ["integer","null"]},    # total blocks per subject, if stated
            "blocks": {"type": "array", "items": block_item_schema},
            "block_order_policy": {"type": "string"},       # "as_described" | "alternating" | "counterbalanced" | "unknown"
            "notes": {"type": "array","items":{"type":"string"}},
            "confidence_overall": {"type": "number"},       # 0..1
            "block_description_text": {"type": "string"}    # concise <= ~200 words textual extraction of block structure
        },
        "required": [
            "block_count","blocks","block_order_policy",
            "notes","confidence_overall","block_description_text"
        ]
    }

    system_prompt = (
        "Extract a conceptual BLOCK STRUCTURE from the experiment description.\n"
        "Output block labels, categories, repeats, and any stated trial counts per block purely from the prose.\n"
        "Also produce a concise textual summary (<= ~200 words) of how blocks are organized (practice/training/main,\n"
        "phases, alternations, counterbalancing, trials per block, etc.). If blocks are not described, output a single\n"
        "'Main' block with low confidence and a brief description stating that no blocks are specified.\n"
        "\n"
        "Guidelines:\n"
        "• Use concise labels (Practice, Training, Main, Speed, Accuracy, Block 1..N, etc.).\n"
        "• Include brief supporting phrases in 'evidence' for each block.\n"
        "• Return JSON matching the schema exactly."
    )

    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=system_prompt,
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMBlocksFromDescription",
        strict_schema=True,
        reasoning_effort="medium",
        text_verbosity="low",
    ) or {}

# ---------- the step ----------

class LLMBlocksFromDescription(BaseStep):
    """
    Step 1/4: derive a conceptual block structure **only** from the paper/materials description,
    and extract a concise textual description of the block plan.

    Input:
      • artifacts/meta/experiments_detailed.json  (from DescribeExperiments)

    Output (combined):
      • artifacts/meta/llm_blocks_from_description.json
        {
          "items": [
            {
              "experiment_title": "...",
              "status": "ok",
              "plan": { ... schema above ... }
            },
            ...
          ]
        }

    Output (per-item):
      • artifacts/llm_blocks_from_description/{idx}.json
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_blocks_from_description",
            artifact="meta/llm_blocks_from_description.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        dst = project.artifacts_dir / self.artifact
        return True if self._force else not dst.exists()

    # mapped; BaseStep aggregates per-item results
    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or item.get("experiment_title") or f"Experiment {idx+1}"
        desc_map = _load_descriptions(project)
        entry = desc_map.get(title) or {}

        summary = (entry.get("standalone_summary") or "").strip()
        aliases = entry.get("aliases") or []

        # Build prompt
        prompt = textwrap.dedent(f"""
        TITLE:
        {title}

        ALIASES (if any):
        {aliases if isinstance(aliases, list) else []}

        EXPERIMENT SUMMARY:
        {summary or "(no summary provided)"}

        TASK:
        Produce a conceptual block structure for this experiment. How many blocks are there per subject and
        how many trials per block? Are there any practice or training blocks? Extract all information from the
        description that would help reconstruct the intended block structure. If no blocks are described,
        create a single conceptual "Main" block and provide a brief note that the paper does not specify blocks.

        Additionally, extract a concise textual description (<= ~200 words) summarizing the block organization.
        """)

        raw = _generate_from_description(prompt) or {}

        # Safe fallback (must match schema)
        if not raw or not isinstance(raw.get("blocks"), list) or len(raw["blocks"]) == 0:
            raw = {
                "block_count": None,
                "blocks": [{
                    "label": "Main",
                    "category": "main",
                    "repeats": None,
                    "n_trials_expected": None,
                    "notes": "Fallback conceptual block; description provides no explicit blocks.",
                    "evidence": [],
                    "confidence": 0.2
                }],
                "block_order_policy": "unknown",
                "notes": ["No explicit blocks found in description."],
                "confidence_overall": 0.2,
                "block_description_text": "No explicit blocks described; treat the session as a single Main block."
            }

        return {
            "experiment_title": title,
            "status": "ok",
            "plan": raw
        }
