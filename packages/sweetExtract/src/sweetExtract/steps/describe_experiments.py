# sweetExtract/steps/describe_experiments.py
from __future__ import annotations
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.steps.extract_studies import ExtractStudies
from sweetExtract.project import Project
from sweetExtract.utils.llm_client import generate_response


class DescribeExperiments(BaseStep):
    """
    Produce a standalone description per experiment.

    - Accepts force=True to re-run this step even if artifacts already exist.
    - Preserves/extends aliases: starts from ExtractStudies.item['aliases'] and, if the
      LLM emits a different title, appends that literal string to aliases.
    """

    def __init__(self, force: bool = False):
        super().__init__(
            name="describe_experiments",
            artifact="meta/experiments_detailed.json",
            depends_on=[ExtractStudies],
            map_over=ExtractStudies,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        # Re-run if forced, otherwise run when upstream artifact exists and ours is missing.
        if self._force:
            return True
        upstream = project.artifacts_dir / "meta" / "studies.json"
        mine     = project.artifacts_dir / "meta" / "experiments_detailed.json"
        return upstream.exists() and not mine.exists()

    def compute(self, project: Project) -> Any:
        raise NotImplementedError

    def compute_one(
        self,
        project: Project,
        item: Dict,
        idx: int,
        all_items: List[Dict],
        prior_outputs: List[Dict],
    ) -> Dict:
        # Schema for the LLM’s structured output (we’ll add aliases after)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "standalone_summary": {"type": "string"},
            },
            "required": ["title", "standalone_summary"],
        }

        # Compact context of earlier experiments to resolve cross-refs
        prior_summary = ""
        if prior_outputs:
            bullets = []
            for j, prev in enumerate(prior_outputs):
                bullets.append(
                    f"- Exp {j + 1} — {prev.get('title', '')}: {(prev.get('standalone_summary') or '')[:300]}"
                )
            prior_summary = "Earlier experiments (for resolving references):\n" + "\n".join(bullets)

        system_prompt = (
            "Produce a replication-ready, standalone description. "
            "If this experiment refers to previous experiments in the paper, inline the needed details so the result "
            "is self-contained. Return ONLY JSON matching the schema."
        )

        user_prompt = (
            f"Target experiment title: {item.get('title', '(unknown)')}\n"
            f"One-sentence summary from paper: {item.get('summary', '')}\n\n"
            f"{prior_summary}\n\n"
            "Write a complete standalone description with concrete details if present."
        )

        out = generate_response(
            model="gpt-5",
            system_prompt=system_prompt,
            prompt=user_prompt,
            file_paths=[project.pdf_path],
            json_schema=schema,
            reasoning_effort="low",
        ) or {}

        # Canonical title the LLM thinks; fallback to upstream title
        llm_title = (out.get("title") or "").strip()
        upstream_title = (item.get("title") or f"Experiment {idx + 1}").strip()
        final_title = llm_title or upstream_title

        # Start aliases from upstream (if any), then append LLM title ONLY if different & non-empty
        aliases: List[str] = []
        if isinstance(item.get("aliases"), list):
            aliases.extend([a for a in item["aliases"] if isinstance(a, str) and a.strip()])

        if llm_title and llm_title not in aliases:
            # Only literal append; no normalization.
            aliases.append(llm_title)

        if upstream_title and upstream_title not in aliases:
            aliases.append(upstream_title)

        return {
            "title": final_title,
            "standalone_summary": out.get("standalone_summary", "").strip(),
            "aliases": aliases,
        }
