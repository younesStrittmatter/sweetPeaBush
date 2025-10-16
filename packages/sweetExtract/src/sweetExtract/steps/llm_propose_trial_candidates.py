# sweetExtract/steps/llm_propose_trial_candidates.py
from __future__ import annotations
import os, json, datetime
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.catalog_for_llm import CatalogForLLM


def _generate(plan_prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {
            "experiment_title": "",
            "decision": "unsure",
            "reason": "LLM unavailable",
            "likely_per_subject_folder": False,
            "per_subject_dir": "",
            "filename_pattern_hint": "",
            "sheets_expected_same": True,
            "candidates": [],
            "diagnostics": {"notes": ["LLM unavailable"], "model": "", "warnings": [], "round": 0},
        }

    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You are a careful data wrangler. Given an experiment description and a slimmed data catalog, "
            "propose file(s) that most likely contain subject–trial-wise rows for the target experiment.\n\n"
            "STRICT RULES:\n"
            "1) Treat file paths and names as strong evidence. Prefer items whose path/name clearly references the target scope.\n"
            "2) If a file is a combined dataset spanning multiple experiments/conditions, you MAY propose it, "
            "   BUT include precise filters using realistic discriminator columns with concrete values.\n"
            "3) Do NOT assign the same specific file to different experiments unless clearly shared; if shared, include per-experiment filters.\n"
            "4) Prefer tabular sources (.csv/.tsv/.xlsx/.json) over per-subject raw folders; directories are allowed only as low-score fallbacks.\n"
            "5) For spreadsheets, choose an existing sheet name if available; otherwise leave sheet empty.\n"
            "6) Provide subject_id_hints and trial_index_hints as likely header names from the catalog—keep them as hints.\n"
            "7) Return a small set (1–3) ordered by confidence with clear reasons.\n"
            "8) ALSO explicitly indicate whether the trial-wise data likely exist as many per-subject files inside ONE directory "
            "(return: likely_per_subject_folder=true/false). If true, propose the most plausible directory path "
            "and an indicative filename pattern (e.g., 'S{num}_*.csv', 'subj_*_trials.xlsx').\n"
            "Return ONLY JSON matching the schema."
        ),
        prompt=plan_prompt,
        file_paths=None,  # do not attach JSON; Responses only supports certain file types
        json_schema=schema,
        schema_name="TrialCandidates",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return out or {}


def _load_catalog_chunks(catalog_llm_path: Path, max_chars_per_chunk: int = 45_000) -> List[str]:
    """
    Read artifacts/meta/catalog_llm.json and produce multiple text chunks so we never truncate.
    Each entry: relpath | sheet | columns_head=[...]
    """
    if not catalog_llm_path.exists():
        return ["CATALOG LLM: (missing)"]

    try:
        obj = json.loads(catalog_llm_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"CATALOG LLM: (failed to read: {type(e).__name__}: {e})"]

    files = obj.get("files") or []

    header = f"CATALOG LLM (files={len(files)}):"
    chunks: List[str] = []
    cur: List[str] = [header]
    cur_len = len(header) + 1

    def flush():
        nonlocal cur, cur_len
        if len(cur) > 1:
            chunks.append("\n".join(cur))
        cur = [header]
        cur_len = len(header) + 1

    for f in files:
        rel = f.get("relpath", "")
        sheet = f.get("sheet", None)
        cols = f.get("columns_head") or []
        cols_str = ", ".join(str(c) for c in cols[:64])
        line = f"- relpath={rel} | sheet={'' if sheet is None else sheet} | columns_head=[{cols_str}]"
        if cur_len + len(line) + 1 > max_chars_per_chunk:
            flush()
        cur.append(line)
        cur_len += len(line) + 1

    flush()
    return chunks or ["CATALOG LLM: (empty)"]


def _append_alias(project: Project, canonical_title: str, alias_title: str, source_step: str) -> None:
    """
    Append an alias for an experiment into artifacts/meta/experiment_aliases.json.
    Idempotent (no duplicates). Very lightweight; can be called by any step.
    """
    if not alias_title or alias_title.strip() == canonical_title.strip():
        return

    path = project.artifacts_dir / "meta" / "experiment_aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"items": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {"items": []}
        except Exception:
            data = {"items": []}

    # find or create record
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = []
        data["items"] = items

    rec = None
    for it in items:
        if isinstance(it, dict) and it.get("experiment_title") == canonical_title:
            rec = it
            break
    if rec is None:
        rec = {"experiment_title": canonical_title, "aliases": []}
        items.append(rec)

    aliases = rec.get("aliases")
    if not isinstance(aliases, list):
        aliases = []
        rec["aliases"] = aliases

    if alias_title not in aliases:
        aliases.append(alias_title)

    # optional simple meta (append-only log)
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta

    updates = meta.get("updates")
    if not isinstance(updates, list):
        updates = []
        meta["updates"] = updates

    updates.append({
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "step": source_step,
        "experiment_title": canonical_title,
        "added_alias": alias_title,
    })

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class LLMProposeTrialCandidates(BaseStep):
    """
    For each experiment, ask the LLM (with chunked catalog_llm text) to propose subject–trial-wise file candidates,
    and explicitly decide if the data are likely per-subject files in one folder.

    Combined artifact: artifacts/meta/llm_propose_trial_candidates.json
    Per-item artifacts: artifacts/llm_propose_trial_candidates/{idx}.json
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_propose_trial_candidates",
            artifact="meta/llm_propose_trial_candidates.json",
            depends_on=[FilterEmpiricalExperiments, CatalogForLLM],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        catalog_llm = project.artifacts_dir / "meta" / "catalog_llm.json"
        out_path = project.artifacts_dir / self.artifact
        return catalog_llm.exists() and (self._force or not out_path.exists())

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError  # mapped step; compute_one is used

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict:
        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experiment_title": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": ["has_trialwise", "needs_transform", "not_trialwise_applicable", "unsure"]
                },
                "reason": {"type": "string"},
                "likely_per_subject_folder": {"type": "boolean"},
                "per_subject_dir": {"type": "string"},
                "filename_pattern_hint": {"type": "string"},
                "sheets_expected_same": {"type": "boolean"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "relpath": {"type": "string"},
                            "sheet": {"type": "string"},
                            "format": {"type": "string", "enum": ["csv","tsv","excel","json","other"]},
                            "score": {"type": "number"},
                            "reason": {"type": "string"},
                            "subject_id_hints": {"type": "array", "items": {"type": "string"}},
                            "trial_index_hints": {"type": "array", "items": {"type": "string"}},
                            "condition_hints": {"type": "array", "items": {"type": "string"}},
                            "filters": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "column": {"type": "string"},
                                        "op": {
                                            "type": "string",
                                            "enum": ["==","!=", "in","not_in","contains","regex",">",">=","<","<="]
                                        },
                                        "value_str": {"type": "string"},
                                        "values": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["column", "op", "value_str", "values"]
                                }
                            }
                        },
                        "required": [
                            "relpath", "sheet", "format", "score", "reason",
                            "subject_id_hints", "trial_index_hints",
                            "condition_hints", "filters"
                        ]
                    }
                },
                "diagnostics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "notes": {"type": "array", "items": {"type": "string"}},
                        "model": {"type": "string"},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "round": {"type": "number"}
                    },
                    "required": ["notes", "model", "warnings", "round"]
                }
            },
            "required": [
                "experiment_title", "decision", "reason", "candidates", "diagnostics",
                "likely_per_subject_folder", "per_subject_dir", "filename_pattern_hint", "sheets_expected_same"
            ]
        }

        title = item.get("title") or f"Experiment {idx + 1}"
        desc = (item.get("standalone_summary") or "").strip()

        catalog_llm_path = project.artifacts_dir / "meta" / "catalog_llm.json"
        chunks = _load_catalog_chunks(catalog_llm_path, max_chars_per_chunk=45_000)

        base_prompt = (
            f"Target experiment: {title}\n\n"
            f"Experiment description (standalone):\n{desc}\n\n"
            "Below is the slimmed catalog (possibly split across multiple parts). "
            "Use file paths, sheet names, and columns_head as strong evidence:\n\n"
        )

        plan_prompt = base_prompt + "\n\n".join(chunks) + (
            "\n\nTask:\n"
            "- Propose 1–3 high-precision candidate files for THIS experiment’s subject–trial-wise data. "
            "Include filters if a file is a combined dataset.\n"
            "- Additionally, decide if the trial-wise data are most likely present as MANY per-subject files inside ONE directory. "
            "If yes, set likely_per_subject_folder=true and propose the best parent directory path (per_subject_dir) "
            "and an indicative filename pattern (filename_pattern_hint). If unsure, set false and leave those empty.\n"
            "- If Excel, say whether members of a per-subject cluster are expected on the same sheet (sheets_expected_same=true/false).\n"
            "Return ONLY JSON matching the schema."
        )

        out = _generate(plan_prompt, schema)

        # Ensure defaults
        if not isinstance(out, dict):
            out = {}
        out.setdefault("experiment_title", title)
        out.setdefault("decision", "unsure")
        out.setdefault("reason", "")
        out.setdefault("likely_per_subject_folder", False)
        out.setdefault("per_subject_dir", "")
        out.setdefault("filename_pattern_hint", "")
        out.setdefault("sheets_expected_same", True)
        out.setdefault("candidates", [])
        out.setdefault("diagnostics", {"notes": [], "model": "", "warnings": [], "round": 0})

        # Record alias if the model echoed/normalized a different title.
        model_title = (out.get("experiment_title") or "").strip()
        if model_title and model_title != title:
            _append_alias(project, canonical_title=title, alias_title=model_title, source_step=self.name)

        # Always rewrite the experiment_title to the canonical one for downstream consistency
        out["experiment_title"] = title
        return out
