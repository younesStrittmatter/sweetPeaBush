# sweetExtract/steps/collect_used_stimuli.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_consolidate_timeline import LLMConsolidateTimeline

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
    s = re.sub(r"[^a-zA-Z0-9\\-_.]", "", s)
    return s.lower()[:120] or "exp"

class CollectUsedStimuli(BaseStep):
    """
    Per experiment: read meta/llm_stimuli_consolidation.json and list the *exact*
    SweetBean class names used in the external timeline.

    Output:
      - meta/used_stimuli.json  -> {items:[{experiment_title, slug, stimuli:[...]}]}
      - meta/used_stimuli.report.json (diagnostics)
    """
    artifact_is_list = False
    default_array_key = "items"

    def __init__(self, force: bool=False):
        super().__init__(
            name="collect_used_stimuli",
            artifact="meta/used_stimuli.json",
            depends_on=[LLMConsolidateTimeline, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    def _consolidation_map(self, project: Project) -> Dict[str, Dict[str, Any]]:
        obj = _read_json(project.artifacts_dir / "meta" / "llm_stimuli_consolidation.json") or {}
        items = obj.get("items") or ([obj] if obj.get("experiment_title") else [])
        return {(it.get("experiment_title") or it.get("title") or ""): it for it in items}

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior: List[Dict]) -> Dict[str, Any]:
        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)
        cons  = self._consolidation_map(project).get(title) or {}
        tl    = cons.get("timeline") or []
        stimuli = sorted({(u or {}).get("stimulus") for u in tl if (u or {}).get("stimulus")})
        return {"experiment_title": title, "slug": slug, "stimuli": stimuli}

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        used_path = project.artifacts_dir / self.artifact
        _write_json(used_path, {"items": results})

        # small diagnostics file: what we saw per experiment
        diag = { (it["slug"]): {"title": it["experiment_title"], "stimuli": it.get("stimuli", [])}
                 for it in results }
        _write_json(project.artifacts_dir / "meta" / "used_stimuli.report.json", diag)
        return {"items": results, "artifact": str(used_path.relative_to(project.artifacts_dir))}
