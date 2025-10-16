# sweetExtract/steps/sb_trial_schema_for_llm.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.sb_schema_preview import SBTrialSchemaPreview


_ALIAS_FILES = ("meta/exp_alias_index.json", "meta/experiment_aliases.json")


def _all_equal_lists(lists: List[List[str]]) -> bool:
    if not lists:
        return True
    first = lists[0]
    return all(lst == first for lst in lists[1:])


def _most_common_order(lists: List[List[str]]) -> List[str]:
    freq = Counter()
    first_pos: Dict[str, int] = {}
    seen_any = False
    for headers in lists:
        seen_any = True
        for i, h in enumerate(headers):
            freq[h] += 1
            if h not in first_pos:
                first_pos[h] = i
    if not seen_any:
        return []
    cols = list(freq.keys())
    cols.sort(key=lambda c: (-freq[c], first_pos.get(c, 10_000)))
    return cols


def _unique_preserve_order(values: List[str]) -> List[str]:
    out, seen = [], set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


class SBTrialSchemaForLLM(BaseStep):
    """
    Collapse the per-subject preview into a single LLM-facing schema per experiment:

      - headers: a single header list
          * if all files share identical headers -> use that list
          * else -> merged list ordered by frequency, then first-seen position

      - columns: { col -> { "samples": [up to N unique values],
                            "sample_truncated": bool,
                            "n_unique": int (from preview if available) } }
    """

    def __init__(self, max_samples_per_col: int = 10, force: bool = False):
        super().__init__(
            name="sb_trial_schema_for_llm",
            artifact="meta/sb_trial_schema_for_llm.json",
            depends_on=[SBTrialSchemaPreview],
            map_over=FilterEmpiricalExperiments,
        )
        self.max_samples_per_col = max_samples_per_col
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        art = project.artifacts_dir
        preview = art / "meta" / "sb_schema_preview.json"
        out     = art / "meta" / "sb_trial_schema_for_llm.json"
        if not preview.exists():
            return False
        if self._force:
            return True
        return not out.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    # ---------- alias + wiring helpers ----------

    def _load_trials(self, project: Project) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        art = project.artifacts_dir
        obj = json.loads((art / "meta" / "trials_index.json").read_text(encoding="utf-8"))
        items: List[Dict[str, Any]] = obj.get("items") or []
        canon_to_idx: Dict[str, int] = {}
        for i, it in enumerate(items):
            t = it.get("experiment_title")
            if isinstance(t, str) and t:
                canon_to_idx[t] = i
        return items, canon_to_idx

    def _alias_to_idx(self, project: Project) -> Dict[str, int]:
        art = project.artifacts_dir
        trials_items, canon_to_idx = self._load_trials(project)

        alias_map: Dict[str, int] = {}
        for title, i in canon_to_idx.items():
            alias_map.setdefault(title, i)
        for i, it in enumerate(trials_items):
            for a in (it.get("aliases") or []):
                if isinstance(a, str) and a:
                    alias_map.setdefault(a, i)

        for rel in _ALIAS_FILES:
            p = art / rel
            if not p.exists():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(obj.get("index"), dict):
                for alias, rec in obj["index"].items():
                    if not isinstance(alias, str) or not alias:
                        continue
                    i = rec.get("idx")
                    if not isinstance(i, int):
                        et = rec.get("experiment_title")
                        i = canon_to_idx.get(et) if isinstance(et, str) else None
                    if isinstance(i, int):
                        alias_map[alias] = i
                continue

            if isinstance(obj.get("items"), list):
                for rec in obj["items"]:
                    if not isinstance(rec, dict):
                        continue
                    i = rec.get("idx")
                    if not isinstance(i, int):
                        et = rec.get("experiment_title")
                        i = canon_to_idx.get(et) if isinstance(et, str) else None
                    if not isinstance(i, int):
                        continue
                    for a in (rec.get("aliases") or []):
                        if isinstance(a, str) and a:
                            alias_map[a] = i

        return alias_map

    def _load_preview_items(self, project: Project) -> List[Dict[str, Any]]:
        p = project.artifacts_dir / "meta" / "sb_schema_preview.json"
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return (obj.get("items") or []) if isinstance(obj, dict) else []
        except Exception:
            return []

    def _find_preview_for_title_or_alias(self, project: Project, title: str) -> Optional[Dict[str, Any]]:
        previews = self._load_preview_items(project)

        for it in previews:
            if (it or {}).get("experiment_title") == title and (it or {}).get("status") == "ok":
                return it

        alias_to_idx = self._alias_to_idx(project)
        if title not in alias_to_idx:
            return None

        trials_items, _canon = self._load_trials(project)
        i = alias_to_idx[title]
        if not (0 <= i < len(trials_items)):
            return None
        dest_dir = (trials_items[i] or {}).get("dest_dir") or ""
        if not isinstance(dest_dir, str) or not dest_dir:
            return None

        prefix = dest_dir.rstrip("/") + "/"
        for it in previews:
            if (it or {}).get("status") != "ok":
                continue
            subj_files = (it.get("subject_files") or [])
            pfh_keys = list((it.get("per_file_headers") or {}).keys())
            pool = [*subj_files, *pfh_keys]
            if any(isinstance(p, str) and (p == dest_dir or p.startswith(prefix)) for p in pool):
                return it

        return None

    # ---------- per-item ----------

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or f"Experiment {idx+1}"
        prev = self._find_preview_for_title_or_alias(project, title)
        if not prev:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_preview_for_title_or_alias",
            }

        pfh: Dict[str, List[str]] = prev.get("per_file_headers") or {}
        header_lists = [h for h in pfh.values() if isinstance(h, list)]
        same_headers = _all_equal_lists(header_lists)

        if same_headers and header_lists:
            headers = header_lists[0]
            headers_from = "identical"
        else:
            headers = _most_common_order(header_lists)
            headers_from = "merged"

        cols_meta: Dict[str, Any] = prev.get("columns") or {}
        columns_out: Dict[str, Dict[str, Any]] = {}

        ordered_cols = list(headers)
        for c in cols_meta.keys():
            if c not in ordered_cols:
                ordered_cols.append(c)

        for col in ordered_cols:
            info = cols_meta.get(col, {}) or {}
            examples = info.get("examples") or []
            uniq = _unique_preserve_order([str(x) for x in examples])
            sample = uniq[: self.max_samples_per_col]

            # NEW: propagate n_unique if present; use it for truncation
            n_unique = info.get("n_unique")
            if isinstance(n_unique, int):
                sample_truncated = n_unique > len(sample)
            else:
                sample_truncated = len(uniq) > len(sample)

            columns_out[col] = {
                "samples": sample,
                "sample_truncated": sample_truncated,
                "n_unique": int(n_unique) if isinstance(n_unique, int) else None,
            }

        return {
            "experiment_title": title,
            "status": "ok",
            "headers": headers,
            "columns": columns_out,
            "same_headers": same_headers,
            "headers_from": headers_from,
        }
