from __future__ import annotations
import os
import textwrap
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments

def _llm_classify_empirical(prompt: str) -> Dict[str, Any]:
    """
    Ask the LLM whether the described study collects NEW experimental data.
    Returns a strictly-typed JSON object with is_empirical + minimal fields;
    we won't store these results, just use to filter.
    """
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_empirical": {"type": "boolean"},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["is_empirical", "reason", "confidence"],
    }

    system_prompt = (
        "Decide whether the described study collects NEW experimental data (empirical), "
        "or is purely re-analysis/simulation/theory without new collection.\n\n"
        "Be conservative: only mark is_empirical=True when the text clearly indicates NEW data collection for THIS study.\n"
        "Return JSON matching the schema exactly."
    )

    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=system_prompt,
        prompt=prompt,
        json_schema=schema,
        schema_name="EmpiricalExperimentDecisionMinimal",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {}

class FilterEmpiricalExperiments(BaseStep):
    """
    Minimal filter:
      - Loads detailed experiment descriptions (DescribeExperiments output).
      - Uses LLM to decide if each item is empirical (NEW data).
      - Writes ONLY the original items for empirical studies to:
            artifacts/meta/experiments_empirical_detailed.json
        with shape:
            { "items": [ ...subset of original DescribeExperiments items... ] }

    No flags, no per-item side artifacts — just picking, not adding.
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="filter_empirical_experiments",
            artifact="meta/experiments_empirical_detailed.json",
            depends_on=[DescribeExperiments],
            map_over=None,  # we do the loop ourselves to return a single filtered file
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        # Load all descriptions + alias index via the Project helper
        idx = project.load_experiment_descriptions()
        items: List[Dict[str, Any]] = idx["items"]  # full set

        kept: List[Dict[str, Any]] = []
        for it in items:
            title = (it.get("title") or it.get("experiment_title") or "").strip()
            summary = (it.get("standalone_summary") or "").strip()

            # If there's no standalone description, we can't make a confident call → skip
            if not summary:
                continue

            user_prompt = textwrap.dedent(f"""
            TITLE: {title or "(untitled)"}

            STANDALONE DESCRIPTION:
            {summary}

            TASK:
            Decide if THIS study collects NEW experimental data. If it simply reuses earlier data, fits models,
            or runs simulations without collecting new observations, mark it as non-empirical.
            """)

            decision = _llm_classify_empirical(user_prompt) or {}
            if bool(decision.get("is_empirical")):
                # Keep the original item as-is (don't add flags/keys).
                kept.append(it)

        # Return a “DescribeExperiments-shaped” JSON with only empirical items.
        return {"items": kept}
