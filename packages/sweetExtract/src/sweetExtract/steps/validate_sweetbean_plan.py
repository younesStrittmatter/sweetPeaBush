# sweetExtract/steps/validate_sweetbean_plan.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments
from sweetExtract.steps.sb_schema_preview import SBTrialSchemaPreview
from sweetExtract.steps.llm_build_sweetbean_plan import LLMBuildSweetBeanPlan


class ValidateSweetBeanPlan(BaseStep):
    """
    Validate that SweetBean plan only references columns seen in the schema preview.
    Combined: artifacts/sweetbean/plan_validation.json
    Per-item: artifacts/sweetbean/plan_validation/{idx}.json
    """

    def __init__(self):
        super().__init__(
            name="validate_sweetbean_plan",
            artifact="sweetbean/plan_validation.json",
            depends_on=[DescribeExperiments, SBTrialSchemaPreview, LLMBuildSweetBeanPlan],
            map_over=DescribeExperiments,
        )

    def should_run(self, project: Project) -> bool:
        plan_dir = project.artifacts_dir / "sweetbean" / "plan"
        return plan_dir.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError  # mapped

    def _load_for_title(self, project: Project, title: str) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        # preview
        preview_combined = project.artifacts_dir / "sweetbean" / "schema_preview.json"
        plan_combined = project.artifacts_dir / "sweetbean" / "plan.json"
        p_obj = json.loads(preview_combined.read_text(encoding="utf-8")) if preview_combined.exists() else {}
        l_obj = json.loads(plan_combined.read_text(encoding="utf-8")) if plan_combined.exists() else {}
        prev = None
        plan = None
        for it in (p_obj.get("items") or []):
            if it.get("experiment_title") == title and it.get("status") == "ok":
                prev = it
                break
        for it in (l_obj.get("items") or []):
            if it.get("experiment_title") == title:
                plan = it
                break
        return prev, plan

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        title = item.get("title") or f"Experiment {idx+1}"
        preview, plan = self._load_for_title(project, title)

        if not plan:
            return {"experiment_title": title, "status": "skipped", "reason": "no_plan"}

        if not preview:
            return {"experiment_title": title, "status": "skipped", "reason": "no_preview"}

        known_cols = set(preview.get("column_union") or [])
        problems: List[str] = []
        warnings: List[str] = []

        subj_col = plan.get("subject_id_col", "")
        trial_col = plan.get("trial_index_col", "")
        if subj_col and subj_col not in known_cols:
            problems.append(f"subject_id_col '{subj_col}' not in preview columns")
        if trial_col and trial_col not in known_cols:
            warnings.append(f"trial_index_col '{trial_col}' not in preview columns (optional)")

        # Validate row selectors & params
        for b in (plan.get("blocks") or []):
            sel = b.get("row_selector") or []
            for clause in sel:
                col = clause.get("column", "")
                if col and col not in known_cols:
                    problems.append(f"block '{b.get('name','?')}' selector references unknown column '{col}'")
            for st in (b.get("sequence") or []):
                params = st.get("params") or {}
                for pname, binding in params.items():
                    if not isinstance(binding, dict):
                        continue
                    if binding.get("kind") == "column":
                        col = binding.get("value", "")
                        if col and col not in known_cols:
                            problems.append(f"stim '{st.get('type','?')}' param '{pname}' references unknown column '{col}'")

        # Validate param_bindings keys exist
        for col in (plan.get("param_bindings") or {}).keys():
            if col not in known_cols:
                problems.append(f"param_bindings references unknown column '{col}'")

        status = "ok" if not problems else "error"
        return {
            "experiment_title": title,
            "status": status,
            "problems": problems,
            "warnings": warnings,
            "known_columns": sorted(known_cols),
        }
