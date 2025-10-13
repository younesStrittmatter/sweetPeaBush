# packages/sweetExtract/src/sweetExtract/steps/llm_pick_stimuli.py
from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional, Set
from importlib.resources import files

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments
from sweetExtract.steps.llm_infer_paradigm import LLMInferParadigm

# ---------- load palette ----------
_SB_INDEX = json.loads(
    files("sweetExtract").joinpath("info/sb/stimuli_index.json").read_text("utf-8")
)
_PALETTE_TITLES: Set[str] = {
    (s.get("title") or s["key"]).strip() for s in _SB_INDEX if (s.get("title") or s.get("key"))
}

def _palette_text() -> str:
    lines = []
    for s in _SB_INDEX:
        nm = (s.get("title") or s["key"])
        desc = (s.get("description") or "").strip() or "(no description)"
        lines.append(f"- {nm}: {desc}")
    return "\n".join(lines)

# ---------- helpers ----------
def _load_paradigm_item(project: Project, idx: int) -> Dict[str, Any]:
    p = project.artifacts_dir / "artifacts" / "llm_infer_paradigm" / f"{idx}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}

def _load_io_map(artifacts_dir, title: str) -> Dict[str, Any]:
    p = artifacts_dir / "meta" / "llm_experiment_io_map.json"
    if not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    for it in obj.get("items") or []:
        if (it or {}).get("experiment_title") == title and (it or {}).get("status") == "ok":
            return it
    return {}

def _generate_json(prompt: str) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "presentation": {"type": "array", "items": {"type": "string"}},
            "response": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["presentation", "response", "notes"]
    }
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "Choose SweetBean stimuli by NAME from the palette. "
            "Include everything needed for a full trial timeline: "
            "main task stimulus(es), plus Fixation, Cue/Sample, Response prompt, Feedback, Blank(ITI) "
            "if appropriate for THIS experiment. "
            "Return JSON ONLY with fields presentation[], response[], notes[]. "
            "Use EXACT names from the palette; do not invent or rename."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMPickStimuli",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {}

# ---------- the Step ----------
class LLMPickStimuli(BaseStep):
    """
    LLM-only selection of SweetBean stimuli (no heuristics).
    Aggregated output: meta/llm_sb_choices.json
    Per-item (auto by BaseStep): artifacts/llm_pick_stimuli/{idx}.json
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_pick_stimuli",
            artifact="meta/llm_sb_choices.json",
            depends_on=[DescribeExperiments, LLMInferParadigm],
            map_over=DescribeExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        return True if self._force else not self.default_artifact(project).exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(
        self,
        project: Project,
        item: Dict,
        idx: int,
        all_items: List[Dict],
        prior_outputs: List[Dict],
    ) -> Dict[str, Any]:
        title = item.get("title") or f"Experiment {idx+1}"
        summary = (item.get("standalone_summary") or "").strip()
        roles   = (_load_paradigm_item(project, idx).get("timeline_roles") or [])
        io      = _load_io_map(project.artifacts_dir, title)

        pres = io.get("presentation") or []
        resp = io.get("response") or []
        pres_cols = sorted({c for g in pres for c in (g.get("columns") or [])})
        resp_cols = sorted({c for g in resp for c in (g.get("columns") or [])})

        prompt = (
            f"TITLE: {title}\n\n"
            f"SUMMARY:\n{summary}\n\n"
            f"PARADIGM ROLES: {roles}\n\n"
            "DATA COLUMNS:\n"
            "  PRESENTATION: " + ", ".join(pres_cols) + "\n"
            "  RESPONSE: "     + ", ".join(resp_cols) + "\n\n"
            "PALETTE (use these exact names):\n"
            + _palette_text() + "\n\n"
            "TASK:\n"
            "- Pick ALL stimuli needed for this experiment's trial timeline.\n"
            "- Put trial screens (including fixation/cue/feedback/blank if applicable) in presentation[];\n"
            "  put the response collector in response[].\n"
            "- Be minimal but complete. Return JSON only."
        )

        raw = _generate_json(prompt) or {"presentation": [], "response": [], "notes": []}

        # Keep only exact palette names and dedupe (no auto-adding / heuristics)
        def _filter_names(xs: List[str]) -> List[str]:
            seen: Set[str] = set()
            out: List[str] = []
            for x in xs or []:
                x = (x or "").strip()
                if x and x in _PALETTE_TITLES and x not in seen:
                    out.append(x); seen.add(x)
            return out

        out = {
            "experiment_title": title,
            "presentation": _filter_names(raw.get("presentation")),
            "response": _filter_names(raw.get("response")),
            "notes": raw.get("notes") or []
        }
        return out
