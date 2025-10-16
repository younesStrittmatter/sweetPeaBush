# sweetExtract/steps/filter_sb_param_inventory_for_experiment.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List, Set

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.extract_sb_param_inventory import ExtractSBParamInventory
from sweetExtract.steps.collect_used_stimuli import CollectUsedStimuli

# --------- io helpers ----------
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
    s = re.sub(r"[^a-zA-Z0-9\-_.]", "", s)
    return s.lower()[:120] or "exp"

class FilterSBParamInventoryForExperiment(BaseStep):
    """
    Intersect the global SB param inventory with each experiment's used stimuli (exact names).
    Writes a SINGLE index file:
      artifacts/meta/sb_param_inventory.filtered_index.json
    with structure:
      {
        "items": [
          {
            "experiment_title": "...",
            "slug": "...",
            "n_classes": 4,
            "items": [ <filtered init_docs entries> ],
            "missing": [ <used names not found in inventory> ]
          }, ...
        ]
      }
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="filter_sb_param_inventory_for_experiment",
            artifact="meta/sb_param_inventory_filtered_index.json",
            depends_on=[ExtractSBParamInventory, CollectUsedStimuli, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    def _global_inventory(self, project: Project) -> List[Dict[str, Any]]:
        obj = _read_json(project.artifacts_dir / "meta" / "sb_param_inventory.json") or {}
        return obj.get("items") or []

    def _used_index(self, project: Project) -> Dict[str, Dict[str, Any]]:
        """Key by exact experiment_title for consistency with CollectUsedStimuli."""
        obj = _read_json(project.artifacts_dir / "meta" / "used_stimuli.json") or {}
        items = obj.get("items") or []
        return {(it.get("experiment_title") or it.get("title") or ""): it for it in items}

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], _prior: List[Dict]) -> Dict[str, Any]:
        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        used_map = self._used_index(project)
        used_rec = used_map.get(title) or {}
        used: Set[str] = set((used_rec.get("stimuli") or []))

        inv = self._global_inventory(project)
        inv_by_name = { (r.get("class_name") or ""): r for r in inv if r.get("class_name") }

        filtered = [inv_by_name[n] for n in used if n in inv_by_name]
        missing  = sorted(list(used - set(inv_by_name.keys())))

        return {
            "experiment_title": title,
            "slug": slug,
            "n_classes": len(filtered),
            "items": filtered,
            "missing": missing
        }

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        out_path = project.artifacts_dir / self.artifact
        _write_json(out_path, {"items": results})
        return {"items": results, "index": str(out_path.relative_to(project.artifacts_dir))}
