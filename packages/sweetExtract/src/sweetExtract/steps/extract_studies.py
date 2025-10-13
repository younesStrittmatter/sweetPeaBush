# sweetExtract/steps/extract_studies.py
from __future__ import annotations
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.utils.llm_client import generate_response


class ExtractStudies(BaseStep):
    """
    Extract study/experiment entries from the paper PDF.

    - Forceable: pass force=True to re-run even if should_run() would skip.
    - Seeds per-study aliases with the canonical title returned here:
        each item -> {"title": "...", "summary": "...", "aliases": ["..."]}

    Artifact shape (unchanged at top level):
      {
        "studies": [
          {"title": str, "summary": str, "aliases": [str, ...]},
          ...
        ]
      }
    """

    artifact_is_list = False
    default_array_key = "studies"

    def __init__(self, force: bool = False):
        super().__init__(
            name="extract_studies",
            artifact="meta/studies.json",
            depends_on=[],
            map_over=None,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        # Run if we have data (normal path) or if explicitly forced.
        return self._force or project.has_data_raw_files()

    def compute(self, project: Project) -> Any:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "studies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"}
                        },
                        "required": ["title", "summary"]
                    }
                }
            },
            "required": ["studies"]
        }

        system_prompt = "Extract all studies/experiments with a short summary."
        user_prompt = "List all studies/experiments in this paper with a one-sentence summary each."

        out = generate_response(
            model="gpt-5",
            system_prompt=system_prompt,
            prompt=user_prompt,
            file_paths=[project.pdf_path],
            json_schema=schema,
            reasoning_effort="low",
        ) or {"studies": []}

        # Post-process: seed aliases with the returned title
        studies_in: List[Dict[str, str]] = out.get("studies") or []
        studies_out: List[Dict[str, Any]] = []
        for s in studies_in:
            title = (s.get("title") or "").strip()
            summary = (s.get("summary") or "").strip()
            # Ensure minimal shape and add aliases
            studies_out.append({
                "title": title or "Untitled experiment",
                "summary": summary,
                "aliases": [title] if title else [],  # initial alias seed
            })

        return {"studies": studies_out}
