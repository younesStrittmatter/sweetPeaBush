# sweetExtract/steps/sb_schema_preview.py
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter, defaultdict

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments


def _peek_csv_headers(path: Path, encoding: str = "utf-8") -> List[str]:
    try:
        with path.open("r", encoding=encoding, errors="ignore", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, [])
            return [h.strip() for h in first if isinstance(h, str)]
    except Exception:
        return []


def _sample_csv_rows(
    path: Path,
    headers: List[str],
    max_rows: int = 50,
    encoding: str = "utf-8",
) -> List[List[str]]:
    rows: List[List[str]] = []
    if not headers:
        return rows
    try:
        with path.open("r", encoding=encoding, errors="ignore", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header
            for i, row in enumerate(reader):
                rows.append(row)
                if i + 1 >= max_rows:
                    break
    except Exception:
        pass
    return rows


def _infer_dtype(samples: List[str]) -> str:
    vals = [s.strip() for s in samples if isinstance(s, str)]
    low = {v.lower() for v in vals}
    if low and low.issubset({"true", "false", "t", "f", "0", "1", "yes", "no"}):
        return "bool"
    try:
        if vals and all(v.replace("-", "", 1).isdigit() for v in vals if v):
            return "int"
    except Exception:
        pass

    def _is_float(x: str) -> bool:
        try:
            float(x.replace(",", "."))
            return True
        except Exception:
            return False

    if vals and all(_is_float(v) for v in vals if v):
        return "float"
    uniq = {v for v in vals if v != ""}
    if 0 < len(uniq) <= 10:
        return "categorical"
    return "text"


class SBTrialSchemaPreview(BaseStep):
    """
    Preview subject-level CSV schemas for each experiment using ONLY prior artifacts.

    Inputs (must exist):
      • artifacts/meta/trials_index.json              (from ExecuteTrialSplit; contains per-experiment subjects + aliases)
      • (optional) artifacts/meta/experiment_aliases.json
          {
            "items": [
              { "experiment_title": "...", "aliases": ["...", ...] },
              ...
            ]
          }
      • (optional) artifacts/meta/exp_alias_index.json
          {
            "index": {
              "<alias>": { "idx": <int>, "experiment_title": "...", "slug": "...", "dest_dir": "..." },
              ...
            }
          }

    Matching is STRICT (exact string match). No normalization.

    Output:
      • combined: artifacts/meta/sb_schema_preview.json
      • per-item: artifacts/sb_schema_preview/{idx}.json

    Use SBTrialSchemaForLLM afterwards to collapse to a single canonical header list.
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(
        self,
        max_subject_files: int = 8,
        max_rows_per_file: int = 50,
        max_examples_per_col: int = 12,
        force: bool = False,
    ):
        super().__init__(
            name="sb_schema_preview",
            artifact="meta/sb_schema_preview.json",
            depends_on=[DescribeExperiments],  # map index; file preconditions enforced in should_run
            map_over=DescribeExperiments,
        )
        self.max_subject_files = max_subject_files
        self.max_rows_per_file = max_rows_per_file
        self.max_examples_per_col = max_examples_per_col
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        art = project.artifacts_dir
        trials_idx = art / "meta" / "trials_index.json"
        out = art / "meta" / "sb_schema_preview.json"
        # Require trials_index.json; alias files optional. Respect force.
        return trials_idx.exists() and (self._force or not out.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        # Mapped step; BaseStep aggregates per-item outputs automatically.
        raise NotImplementedError

    # ---- alias utilities -----------------------------------------------------

    def _load_trials(self, project: Project) -> List[Dict[str, Any]]:
        art = project.artifacts_dir
        obj = json.loads((art / "meta" / "trials_index.json").read_text(encoding="utf-8"))
        return obj.get("items") or []

    def _alias_map_from_trials(self, trials_items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Exact alias -> index, using titles and the 'aliases' arrays in trials_index.json."""
        alias_to_idx: Dict[str, int] = {}
        for i, it in enumerate(trials_items):
            title = it.get("experiment_title")
            if isinstance(title, str) and title and title not in alias_to_idx:
                alias_to_idx[title] = i
            for a in (it.get("aliases") or []):
                if isinstance(a, str) and a and a not in alias_to_idx:
                    alias_to_idx[a] = i
        return alias_to_idx

    def _alias_map_from_experiment_aliases(
        self,
        project: Project,
        trials_items: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Exact alias -> index, based on experiment_aliases.json, only if canonical title exists in trials."""
        art = project.artifacts_dir
        path = art / "meta" / "experiment_aliases.json"
        if not path.exists():
            return {}

        # Build canonical title -> idx for validation
        canon_to_idx: Dict[str, int] = {}
        for i, it in enumerate(trials_items):
            t = it.get("experiment_title")
            if isinstance(t, str) and t:
                canon_to_idx[t] = i

        res: Dict[str, int] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for rec in payload.get("items", []):
                canon = rec.get("experiment_title")
                if not isinstance(canon, str) or canon not in canon_to_idx:
                    continue
                idx = canon_to_idx[canon]
                for a in rec.get("aliases", []) or []:
                    if isinstance(a, str) and a and a not in res:
                        res[a] = idx
        except Exception:
            pass
        return res

    def _alias_map_from_index_file(self, project: Project, n_trials: int) -> Dict[str, int]:
        """Exact alias -> index, based on exp_alias_index.json (if present)."""
        art = project.artifacts_dir
        path = art / "meta" / "exp_alias_index.json"
        if not path.exists():
            return {}
        res: Dict[str, int] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            idx_map = payload.get("index") or {}
            if isinstance(idx_map, dict):
                for alias, rec in idx_map.items():
                    if not isinstance(alias, str) or not alias:
                        continue
                    try:
                        i = int((rec or {}).get("idx"))
                    except Exception:
                        continue
                    if 0 <= i < n_trials and alias not in res:
                        res[alias] = i
        except Exception:
            pass
        return res

    def _build_strict_alias_lookup(self, project: Project) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Build alias->idx with STRICT exact-string keys, combining (in priority order):
          1) titles + aliases from trials_index.json
          2) experiment_aliases.json
          3) exp_alias_index.json
        Earlier sources win on collisions.
        """
        trials_items = self._load_trials(project)
        alias_map = self._alias_map_from_trials(trials_items)
        # merge in experiment_aliases.json (do not overwrite existing keys)
        addl = self._alias_map_from_experiment_aliases(project, trials_items)
        for k, v in addl.items():
            alias_map.setdefault(k, v)
        # merge in old exp_alias_index.json (do not overwrite existing keys)
        more = self._alias_map_from_index_file(project, len(trials_items))
        for k, v in more.items():
            alias_map.setdefault(k, v)
        return trials_items, alias_map

    # ---- per-item ------------------------------------------------------------

    def compute_one(
        self,
        project: Project,
        item: Dict,
        idx: int,
        all_items: List[Dict],
        prior_outputs: List[Dict],
    ) -> Dict:
        title = item.get("title") or f"Experiment {idx+1}"

        trials_items, alias_to_idx = self._build_strict_alias_lookup(project)

        if title not in alias_to_idx:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_preview_for_title_or_alias",
                "subject_files": [],
                "per_file_headers": {},
                "column_union": [],
                "columns": {},
                "notes": [],
            }

        t_idx = alias_to_idx[title]
        if not (0 <= t_idx < len(trials_items)):
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "alias_points_to_invalid_index",
                "subject_files": [],
                "per_file_headers": {},
                "column_union": [],
                "columns": {},
                "notes": [],
            }

        t_item = trials_items[t_idx]
        subjects = [s for s in (t_item.get("subjects") or []) if isinstance(s, dict) and s.get("relpath")]
        if not subjects:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_subject_csvs_in_trials_item",
                "subject_files": [],
                "per_file_headers": {},
                "column_union": [],
                "columns": {},
                "notes": [],
            }

        art_res = project.artifacts_dir.resolve()
        subject_csvs: List[Path] = []
        for s in subjects[: self.max_subject_files]:
            p = (art_res / s["relpath"]).resolve()
            if p.exists() and p.suffix.lower() == ".csv":
                subject_csvs.append(p)

        if not subject_csvs:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_existing_csv_paths_in_trials_item",
                "subject_files": [],
                "per_file_headers": {},
                "column_union": [],
                "columns": {},
                "notes": [],
            }

        # relative path helper (avoid absolute/relative mismatch)
        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(art_res))
            except Exception:
                # Fallback: best-effort string
                return str(p)

        col_examples: Dict[str, List[str]] = defaultdict(list)
        col_seen_in_files: Dict[str, int] = Counter()
        per_file_headers: Dict[str, List[str]] = {}

        for p in subject_csvs:
            headers = _peek_csv_headers(p)
            per_file_headers[_rel(p)] = headers
            for h in headers:
                col_seen_in_files[h] += 1

            rows = _sample_csv_rows(p, headers, max_rows=self.max_rows_per_file)
            for row in rows:
                for h, v in zip(headers, row):
                    if len(col_examples[h]) < self.max_examples_per_col and isinstance(v, str):
                        v = v.strip()
                        if v != "":
                            col_examples[h].append(v)

        column_union = sorted(col_seen_in_files.keys())
        columns: Dict[str, Dict[str, Any]] = {}
        for h in column_union:
            examples = col_examples.get(h, [])
            dtype = _infer_dtype(examples[: min(6, len(examples))])
            columns[h] = {
                "dtype_guess": dtype,
                "examples": examples[: self.max_examples_per_col],
                "seen_in_n_subject_files": int(col_seen_in_files[h]),
            }

        return {
            "experiment_title": title,
            "status": "ok",
            "reason": "",
            "subject_files": [_rel(p) for p in subject_csvs],
            "per_file_headers": per_file_headers,
            "column_union": column_union,
            "columns": columns,
            "notes": [
                f"Scanned up to {self.max_subject_files} subject CSVs and {self.max_rows_per_file} rows/file.",
                "Matched by exact title/alias from trials_index.json and optional alias files; no normalization used.",
            ],
        }
