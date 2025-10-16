# sweetExtract/steps/consolidate_blocks.py
from __future__ import annotations
import json, textwrap, re
from typing import Any, Dict, List, Optional, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.llm_blocks_from_data import LLMBlocksFromData
from sweetExtract.steps.llm_blocks_from_description import LLMBlocksFromDescription
from sweetExtract.steps.sb_trial_schema_for_llm import SBTrialSchemaForLLM

def _read_json(path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_json_map(project: Project, relpath: str, title_key: str) -> Dict[str, Dict[str, Any]]:
    p = project.artifacts_dir / relpath
    obj = _read_json(p) or {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj.get("items"), list):
        for it in obj["items"]:
            title = (it or {}).get(title_key) or (it or {}).get("title")
            if isinstance(title, str) and title:
                out[title] = it
    return out

def _load_data_plans(project: Project) -> Dict[str, Dict[str, Any]]:
    return _load_json_map(project, "meta/llm_blocks_from_data.json", "experiment_title")

def _load_desc_plans(project: Project) -> Dict[str, Dict[str, Any]]:
    return _load_json_map(project, "meta/llm_blocks_from_description.json", "experiment_title")

def _load_trial_schema(project: Project) -> Dict[str, Dict[str, Any]]:
    for cand in ["meta/sb_trial_schema_for_llm.json", "meta/sb_trial_schema_llm.json"]:
        m = _load_json_map(project, cand, "experiment_title")
        if m:
            return m
    return {}

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

_BLOCK_CUES = [
    "block","blk","blockid","blocklabel","trialblock","experimentblock","expblock","runblock",
    "session","sess","run","phase","part","half","segment","epoch","round","series","cycle","set",
    "stage","section","period","main","training","test",
]

def _name_cue_score(name: str) -> int:
    nk = _norm_key(name)
    score = 1000
    for i, cue in enumerate(_BLOCK_CUES):
        if cue in nk:
            score = min(score, i)
    return score

def _column_n_unique(schema_item: Dict[str, Any], col: str) -> Optional[int]:
    cols = (schema_item or {}).get("columns") or {}
    return (cols.get(col) or {}).get("n_unique")

def _score_data_plan(data_item: Dict[str, Any],
                     desc_item: Dict[str, Any],
                     schema_item: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Return (score, reasons). Higher is better."""
    reasons: List[str] = []
    score = 0

    plan = (data_item or {}).get("block_plan") or {}
    split_by = plan.get("split_by") or {}
    split_col = split_by.get("column")

    desc_plan = (desc_item or {}).get("plan") or {}
    desc_block_count = desc_plan.get("block_count")

    if isinstance(split_col, str):
        score += 2
        reasons.append(f"split_by column '{split_col}' present")
        ns = _name_cue_score(split_col)
        if ns < 6:
            score += 2
            reasons.append(f"name suggests block-like column ({split_col})")
        nuniq = _column_n_unique(schema_item, split_col)
        if isinstance(nuniq, int):
            # within-session chunking can have many levels; don't penalize
            if 2 <= nuniq <= 200:
                score += 2
                reasons.append(f"reasonable cardinality n_unique={nuniq}")
        if isinstance(desc_block_count, int) and isinstance(nuniq, int):
            if desc_block_count >= 8 and nuniq <= 3:
                score -= 3
                reasons.append(f"mismatch: description {desc_block_count} blocks vs data {nuniq}")
            elif desc_block_count == nuniq:
                score += 3
                reasons.append("data split count matches description")
    if plan.get("blocks"):
        score += 1
        reasons.append("explicit blocks provided")

    return score, reasons

def _derive_index_fallback(desc_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return a dataset-level index split plan or None.
    We assume data are already per-subject. We do not try to infer total rows here;
    the runtime can compute equal contiguous chunks from the actual dataframe length.
    If description provides trials-per-block, use it; otherwise leave block_size=None.
    """
    plan = (desc_item or {}).get("plan") or {}
    N = plan.get("block_count")
    if not isinstance(N, int) or N <= 1:
        return None

    # Prefer a consistent 'n_trials_expected' from description if present
    blocks = (plan or {}).get("blocks") or []
    counts = [b.get("n_trials_expected") for b in blocks if isinstance(b.get("n_trials_expected"), int)]
    trials_expected = counts[0] if counts and all(c == counts[0] for c in counts) else None

    return {
        "per": "dataset",              # <- always dataset-level (per-subject already handled upstream)
        "n_blocks": N,
        "block_size": int(trials_expected) if isinstance(trials_expected, int) else None,
        "remainder_policy": "distribute_front" if trials_expected is None else "none",
        "assumptions": [
            "Data are per-subject already.",
            f"Equal contiguous index split into {N} chunks.",
            ("Block size from description" if trials_expected else "Block size computed at runtime from total rows"),
        ],
    }

class ConsolidateBlocks(BaseStep):
    """
    Consolidate block plans:
      1) Keep the data-driven plan if it scores as reasonable.
      2) Else, if description provides a clear block count, synthesize a dataset-level
         index split into N equal contiguous chunks (optionally using trials-per-block from description).
      3) Else, fall back to a single block.

    Output: artifacts/meta/blocks_consolidated.json
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="consolidate_blocks",
            artifact="meta/blocks_consolidated.json",
            depends_on=[LLMBlocksFromData, LLMBlocksFromDescription, SBTrialSchemaForLLM],
            map_over=LLMBlocksFromData,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self,
                    project: Project,
                    item: Dict[str, Any],
                    idx: int,
                    all_items: List[Dict[str, Any]],
                    prior_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        title = item.get("experiment_title") or item.get("title") or f"Experiment {idx+1}"

        data_map = _load_data_plans(project)
        desc_map = _load_desc_plans(project)
        schema_map = _load_trial_schema(project)

        data_item = data_map.get(title) or {}
        desc_item = desc_map.get(title) or {}
        schema_item = schema_map.get(title) or {}

        data_score, reasons = _score_data_plan(data_item, desc_item, schema_item)

        # >=4 counts as “good / reasonable”
        if data_score >= 4:
            plan = (data_item.get("block_plan") or {}).copy()
            return {
                "experiment_title": title,
                "status": "ok",
                "strategy": "data_split_by" if plan.get("split_by") else "data_blocks",
                "plan": {
                    "split_by": plan.get("split_by") or None,
                    "split_by_index": None,
                    "blocks": plan.get("blocks") or [],
                    "block_order_policy": plan.get("block_order_policy") or "as-is",
                    "block_order_column": plan.get("block_order_column"),
                    "counterbalance": plan.get("counterbalance"),
                    "notes": (plan.get("notes") or []) + [f"Kept data plan (score={data_score}): " + "; ".join(reasons)],
                },
                "diagnostics": {
                    "data_score": data_score,
                    "data_reasons": reasons,
                    "desc_block_count": (desc_item.get("plan") or {}).get("block_count"),
                    "schema_headers": (schema_item or {}).get("headers"),
                },
            }

        # Index fallback (dataset-level; per-subject handled upstream)
        idx_fallback = _derive_index_fallback(desc_item)
        if idx_fallback:
            return {
                "experiment_title": title,
                "status": "ok",
                "strategy": "description_index_fallback",
                "plan": {
                    "split_by": None,
                    "split_by_index": idx_fallback,
                    "blocks": [],
                    "block_order_policy": "as-indexed",
                    "block_order_column": None,
                    "counterbalance": None,
                    "notes": [
                        "Data/description did not yield a strong column-based plan; using dataset-level equal index split guided by description."
                    ],
                },
                "diagnostics": {
                    "data_score": data_score,
                    "data_reasons": reasons,
                    "desc_block_count": (desc_item.get("plan") or {}).get("block_count"),
                },
            }

        # Last resort
        return {
            "experiment_title": title,
            "status": "ok",
            "strategy": "single_block",
            "plan": {
                "split_by": None,
                "split_by_index": None,
                "blocks": [
                    {
                        "label": "Main",
                        "criteria": {"where": "true"},
                        "order": None,
                        "repeat": None,
                        "randomize_within": None,
                        "selection": {
                            "where": "true",
                            "first_n": None, "last_n": None, "sample_n": None,
                            "percent": None, "column": None, "distinct": None
                        },
                        "gating": None,
                        "trial_order": None,
                        "trial_order_column": None,
                    }
                ],
                "block_order_policy": "as-is",
                "block_order_column": None,
                "counterbalance": None,
                "notes": [
                    "Fallback to a single block: no reliable column nor description-guided index split found."
                ],
            },
            "diagnostics": {
                "data_score": data_score,
                "data_reasons": reasons,
                "desc_block_count": (desc_item.get("plan") or {}).get("block_count"),
            },
        }
