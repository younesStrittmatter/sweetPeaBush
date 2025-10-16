# sweetExtract/steps/auto_refine_trial_split.py
from __future__ import annotations
import json, os, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

# Orchestrated steps
from sweetExtract.steps.probe_candidate_tables import ProbeCandidateTables
from sweetExtract.steps.llm_refine_trial_plan import LLMRefineTrialPlan
from sweetExtract.steps.execute_trial_split import ExecuteTrialSplit
from sweetExtract.steps.validate_trial_split import ValidateTrialSplit

# ---------- small io utils ----------

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _all_validated(validate_meta: Dict[str, Any]) -> bool:
    if not isinstance(validate_meta, dict):
        return False
    items = validate_meta.get("items")
    if not isinstance(items, list) or not items:
        return False
    return all(bool((it or {}).get("validated")) for it in items)

def _remaining_titles(validate_meta: Dict[str, Any]) -> List[str]:
    items = validate_meta.get("items") if isinstance(validate_meta, dict) else None
    if not isinstance(items, list):
        return []
    return [ (it or {}).get("experiment_title") for it in items if not bool((it or {}).get("validated")) ]

def _assemble_validation_from_items(project: Project) -> Tuple[Dict[str, Any], bool]:
    """
    Fallback: if meta/validate_trial_split.json is missing, try to synthesize it
    from per-item files in artifacts/validate_trial_split/*.json. Returns (meta, success).
    """
    item_dir = project.artifacts_dir / "validate_trial_split"
    if not item_dir.exists():
        return {}, False

    items: List[Dict[str, Any]] = []
    for p in sorted(item_dir.glob("*.json")):
        obj = _read_json(p)
        if isinstance(obj, dict) and obj:
            items.append(obj)

    if not items:
        return {}, False

    meta = {"items": items}
    # Write the synthesized combined meta so downstream tools are consistent.
    meta_path = project.artifacts_dir / "meta" / "validate_trial_split.json"
    _write_json(meta_path, meta)
    return meta, True

def _load_validation_meta(project: Project) -> Tuple[Dict[str, Any], bool, bool]:
    """
    Returns (meta, existed_on_disk, was_synthesized_from_items).
    - existed_on_disk: True if meta file was already present.
    - was_synthesized_from_items: True if we built it from per-item files just now.
    """
    meta_path = project.artifacts_dir / "meta" / "validate_trial_split.json"
    if meta_path.exists():
        meta = _read_json(meta_path) or {}
        return meta, True, False

    meta, ok = _assemble_validation_from_items(project)
    return (meta if ok else {}), False, ok

# ---------- the step ----------

class AutoRefineTrialSplit(BaseStep):
    """
    One-shot orchestrator that loops:
        LLMRefineTrialPlan -> ExecuteTrialSplit -> ValidateTrialSplit
    until all experiments validate or a max round limit is reached.

    Behavior:
      • If a previous validation exists and all experiments are validated, this step SKIPS (unless force=True).
      • Otherwise it runs up to `max_rounds` (default: env TRIAL_SPLIT_MAX_ROUNDS or 3).
      • LLMRefineTrialPlan reads per-experiment hints emitted by ValidateTrialSplit between rounds.

    Inputs (indirect via dependencies / internal calls):
      • meta/llm_propose_trial_candidates.json (from ProbeCandidateTables)
      • meta/candidate_probes.json (from ProbeCandidateTables)

    Outputs:
      • meta/auto_refine_trial_split.json  (summary of the run)
      • Side-effects: writes/overwrites artifacts for the three inner steps each round.
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False, max_rounds: int | None = None):
        super().__init__(
            name="auto_refine_trial_split",
            artifact="meta/auto_refine_trial_split.json",
            depends_on=[ProbeCandidateTables, FilterEmpiricalExperiments],  # ensure probes + index exist first
            map_over=None,
        )
        self._force = bool(force)
        self._max_rounds = int(max_rounds if max_rounds is not None else os.getenv("TRIAL_SPLIT_MAX_ROUNDS", 3))

    def should_run(self, project: Project) -> bool:
        # Skip if we already have a validation with all items validated, unless force=True.
        if self._force:
            return True
        meta, existed, _ = _load_validation_meta(project)
        return not (_all_validated(meta) and existed)

    def compute(self, project: Project) -> Dict[str, Any]:
        # Ensure probe artifacts exist (depends_on already ran ProbeCandidateTables, but be robust)
        art = project.artifacts_dir
        if not (art / "meta" / "candidate_probes.json").exists() or not (art / "meta" / "llm_propose_trial_candidates.json").exists():
            ProbeCandidateTables().run(project)

        summary: Dict[str, Any] = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "rounds": [],
            "validated_all": False,
            "max_rounds": self._max_rounds,
        }

        # Fast-path: already validated?
        current_meta, existed, synthesized = _load_validation_meta(project)
        if existed and _all_validated(current_meta) and not self._force:
            summary["validated_all"] = True
            summary["rounds"].append({
                "round": 0,
                "status": "skipped_already_validated",
                "remaining": [],
                "meta_present": True,
                "meta_synthesized": synthesized,
            })
            _write_json(art / self.artifact, summary)
            return summary

        # Iterative refinement

        for r in range(1, self._max_rounds + 1):
            print(f"\t-- [AutoRefineTrialSplit] Round {r} of {self._max_rounds}")
            # Refine plan (LLM reads prior validation hints automatically)
            print(f"\t - [AutoRefineTrialSplit]   LLMRefineTrialPlan")
            LLMRefineTrialPlan(force=True).run(project)
            # Execute split
            print(f"\t - [AutoRefineTrialSplit]   ExecuteTrialSplit")
            ExecuteTrialSplit(force=True).run(project)
            # Validate results
            print(f"\t - [AutoRefineTrialSplit]   ValidateTrialSplit")
            ValidateTrialSplit(force=True).run(project)

            # Load validation meta (or synthesize it from per-item files if needed)
            meta, existed_after, synthesized_after = _load_validation_meta(project)
            remaining = _remaining_titles(meta)
            all_ok = _all_validated(meta)

            summary["rounds"].append({
                "round": r,
                "status": "validated" if all_ok else "needs_revision",
                "remaining": remaining,
                "meta_present": existed_after or synthesized_after,
                "meta_synthesized": synthesized_after,
            })

            if all_ok:
                print(f"\t-- [AutoRefineTrialSplit] All experiments validated after {r} rounds.")
                summary["validated_all"] = True
                break

        _write_json(art / self.artifact, summary)
        return summary
