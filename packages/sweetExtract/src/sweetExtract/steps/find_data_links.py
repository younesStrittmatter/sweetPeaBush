# sweetExtract/steps/find_data_links.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project


def _ask_llm_sources_from_pdf(
    pdf_path: Path, title: str = "", doi: str = "", paper_url: str = ""
) -> List[Dict[str, Any]]:
    """
    LLM-only: attach the PDF (no local extraction). Returns a list of sources.
    """
    try:
        from sweetExtract.utils.llm_client import generate_response  # your wrapper
    except Exception:
        return []

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "provider": {
                            "type": "string",
                            "description": "osf | github | zenodo | figshare | dataverse | dryad | openneuro | kaggle | other"
                        },
                        "url": {"type": "string"},
                        "canonical_id": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "reason": {"type": ["string", "null"]}
                    },
                    "required": ["provider", "url", "canonical_id", "confidence", "reason"]
                }
            }
        },
        "required": ["sources"]
    }

    system_prompt = (
        "You read the attached research paper PDF. "
        "Return dataset repository links for THIS paper only (not unrelated citations). "
        "Prefer canonical landing pages (OSF project, GitHub repo root, Zenodo record page). "
        "If no dataset is provided, return an empty list."
    )

    user_prompt = (
        "Extract dataset download locations for this paper. "
        "For each, provide: provider, url, canonical_id if obvious, confidence (0-1), and short reason.\n\n"
        f"Paper title (optional): {title}\n"
        f"DOI (optional): {doi}\n"
        f"Paper URL (optional): {paper_url}\n"
    )

    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=system_prompt,
        prompt=user_prompt,
        file_paths=[pdf_path],  # attach the PDF
        json_schema=schema,
        schema_name="DataLinks",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return (out or {}).get("sources", [])


def _normalize_provider(p: str) -> str:
    p = (p or "").strip().lower()
    mp = {
        "osf": "osf", "open science framework": "osf",
        "github": "github",
        "zenodo": "zenodo",
        "figshare": "figshare",
        "dataverse": "dataverse",
        "dryad": "dryad",
        "openneuro": "openneuro",
        "kaggle": "kaggle",
    }
    return mp.get(p, "other")


class FindDataLinks(BaseStep):
    def __init__(self, force: bool = False):
        super().__init__(
            name="find_data_links",
            artifact="meta/data_sources.json",
            depends_on=[],
            map_over=None,
        )
        self._force = force  # <- per-step force toggle

    def should_run(self, project: Project) -> bool:
        # If forced, always run.
        if self._force:
            return True
        # Original behavior: only when raw data is missing.
        return not project.has_data_raw_files()

    def compute(self, project: Project) -> Dict[str, Any]:
        model_name = os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5")
        pdf = project.pdf_path

        # Optional metadata if you have it elsewhere
        title = doi = url = ""

        raw_sources = _ask_llm_sources_from_pdf(pdf, title=title, doi=doi, paper_url=url)

        norm = []
        for s in (raw_sources or [])[:5]:
            norm.append({
                "provider": _normalize_provider(s.get("provider", "")),
                "url": (s.get("url") or "").strip(),
                "canonical_id": (s.get("canonical_id") or None),
                "confidence": float(s.get("confidence") or 0.0),
                "reason": (s.get("reason") or None),
            })

        return {
            "sources": norm,
            "meta": {"llm_model": model_name, "forced": bool(self._force)},
        }
