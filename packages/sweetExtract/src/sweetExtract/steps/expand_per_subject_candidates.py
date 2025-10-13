# sweetExtract/steps/expand_per_subject_candidates.py
from __future__ import annotations
import json, re, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.catalog_for_llm import CatalogForLLM
from sweetExtract.steps.llm_propose_trial_candidates import LLMProposeTrialCandidates
from sweetExtract.steps.describe_experiments import DescribeExperiments

# ---- token/subject detection -------------------------------------------------

SUBJ_WORDS = {"s", "subj", "subject", "participant", "id", "pid"}
SEP_RX = re.compile(r"[^A-Za-z0-9]+")
NUM_TOKEN_RX = re.compile(r"^\d{1,6}$")
ALPHA1_RX = re.compile(r"^[A-Za-z]$")

def _parent(relpath: str) -> str:
    return str(Path(relpath).parent).replace("\\", "/")

def _basename(relpath: str) -> str:
    return Path(relpath).name

def _tokenize(name: str) -> List[str]:
    stem = Path(name).stem
    tokens = [t for t in SEP_RX.split(stem) if t != ""]
    return tokens

def _is_id_like(tok: str, prev_tok: Optional[str], next_tok: Optional[str]) -> bool:
    if tok.lower() in SUBJ_WORDS:
        return False  # the tag itself isn't the id, the neighbor likely is
    if NUM_TOKEN_RX.match(tok) or ALPHA1_RX.match(tok):
        return True
    if prev_tok and prev_tok.lower() in SUBJ_WORDS:
        return True
    if next_tok and next_tok.lower() in SUBJ_WORDS:
        return True
    return False

def _make_skeleton(name: str) -> str:
    """
    Replace id-like tokens ANYWHERE (leading, middle, trailing) with '*'.
    Keep extension to avoid mixing .csv vs .xlsx.
    """
    p = Path(name)
    tokens = _tokenize(p.name)
    skel: List[str] = []
    for i, t in enumerate(tokens):
        prev_tok = tokens[i-1] if i > 0 else None
        next_tok = tokens[i+1] if i+1 < len(tokens) else None
        if _is_id_like(t, prev_tok, next_tok):
            skel.append("*")
        else:
            skel.append(t.lower())
    ext = p.suffix.lower()
    return "_".join(skel) + ext

def _extract_subject_key_from_name(name: str) -> Optional[str]:
    """Pull a subject key from filename by choosing the FIRST id-like token anywhere."""
    tokens = _tokenize(Path(name).name)
    for i, t in enumerate(tokens):
        prev_tok = tokens[i-1] if i > 0 else None
        next_tok = tokens[i+1] if i+1 < len(tokens) else None
        if _is_id_like(t, prev_tok, next_tok):
            return t.lower()
    return None

def _norm_headers(cols: Optional[List[str]]) -> List[str]:
    if not cols:
        return []
    return [str(c).strip().lower() for c in cols if str(c).strip()]

def _headers_jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(_norm_headers(a)), set(_norm_headers(b))
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

# ---- alias helpers -----------------------------------------------------------

def _append_alias(project: Project, canonical_title: str, alias_title: str, source_step: str) -> None:
    """
    Append an alias for an experiment into artifacts/meta/experiment_aliases.json.
    Idempotent (no duplicates).
    """
    alias_title = (alias_title or "").strip()
    canonical_title = (canonical_title or "").strip()
    if not alias_title or alias_title == canonical_title:
        return

    path = project.artifacts_dir / "meta" / "experiment_aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"items": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {"items": []}
        except Exception:
            data = {"items": []}

    items = data.get("items")
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

def _load_alias_map(project: Project) -> Tuple[Dict[str, str], set]:
    """
    Returns (alias_to_canonical, canonical_titles_set).
    Reads:
      - meta/experiments_detailed.json (canonical titles)
      - meta/experiment_aliases.json (aliases accumulated by steps)
    """
    art = project.artifacts_dir
    canonical: set = set()
    alias_to_canonical: Dict[str, str] = {}

    # canonical titles from DescribeExperiments
    exp_path = art / "meta" / "experiments_detailed.json"
    if exp_path.exists():
        try:
            obj = json.loads(exp_path.read_text(encoding="utf-8"))
            items = obj.get("items") or (obj if isinstance(obj, list) else [])
            for it in (items or []):
                t = (it or {}).get("title")
                if t:
                    canonical.add(t)
        except Exception:
            pass

    # aliases (if any)
    alias_path = art / "meta" / "experiment_aliases.json"
    if alias_path.exists():
        try:
            al = json.loads(alias_path.read_text(encoding="utf-8")) or {}
            for rec in (al.get("items") or []):
                ct = (rec or {}).get("experiment_title")
                if not ct:
                    continue
                canonical.add(ct)
                for a in (rec.get("aliases") or []):
                    if a and a not in alias_to_canonical:
                        alias_to_canonical[a] = ct
        except Exception:
            pass

    return alias_to_canonical, canonical

def _canonicalize_title(title: str, alias_to_canonical: Dict[str, str], canonical_set: set) -> Tuple[str, bool]:
    """Return (canonical_title, changed?)."""
    if title in canonical_set:
        return title, False
    if title in alias_to_canonical:
        return alias_to_canonical[title], True
    return title, False

# ---- step --------------------------------------------------------------------

class ExpandPerSubjectCandidates(BaseStep):
    """
    Expand LLM-selected seeds into per-subject groups only when appropriate.

    Conditions (flexible):
      - Prefer candidates where the LLM said likely_per_subject_folder=True.
      - Search only within the same directory (or per_subject_dir override).
      - For Excel, if candidate has a sheet name, restrict to that sheet (or honor sheets_expected_same).
      - Accept members when EITHER filename skeleton matches OR header Jaccard >= 0.50.
      - Accept a group if:
          * (LLM hinted True AND ≥2 members), OR
          * (No hint/False AND ≥3 members AND distinct subject IDs ≥2).
    """

    def __init__(self, force: bool = False):
        super().__init__(
            name="expand_per_subject_candidates",
            artifact="meta/trial_candidate_groups.json",
            depends_on=[CatalogForLLM, LLMProposeTrialCandidates, DescribeExperiments],
            map_over=None,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        art = project.artifacts_dir
        inputs_ready = (art / "meta" / "catalog_llm.json").exists() and (art / "meta" / "llm_propose_trial_candidates.json").exists()
        out_missing = not (art / self.artifact).exists()
        return inputs_ready and (self._force or out_missing)

    def compute(self, project: Project) -> Dict[str, Any]:
        art = project.artifacts_dir
        cat_path = art / "meta" / "catalog_llm.json"
        props_path = art / "meta" / "llm_propose_trial_candidates.json"

        catalog = json.loads(cat_path.read_text(encoding="utf-8"))
        proposals = json.loads(props_path.read_text(encoding="utf-8"))
        items: List[Dict[str, Any]] = proposals.get("items") or (proposals if isinstance(proposals, list) else [])

        files = catalog.get("files") or []

        # Build indices
        by_dir: Dict[str, List[Dict[str, Any]]] = {}
        by_key: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
        for f in files:
            rel = f.get("relpath")
            if not rel:
                continue
            d = _parent(rel)
            by_dir.setdefault(d, []).append(f)
            sheet = f.get("sheet")
            sheet = (sheet if sheet is None else str(sheet))
            by_key[(rel, sheet)] = {
                "relpath": rel,
                "sheet": sheet,  # None or str
                "columns_head": f.get("columns_head") or [],
            }

        # Load alias info and canonical titles
        alias_to_canonical, canonical_set = _load_alias_map(project)

        groups: List[Dict[str, Any]] = []
        total_members = 0

        # thresholds
        J_MIN = 0.50

        for exp_idx, exp in enumerate(items):
            raw_title = exp.get("experiment_title") or f"Experiment {exp_idx + 1}"
            canon_title, changed = _canonicalize_title(raw_title, alias_to_canonical, canonical_set)
            if changed:
                # Record that we observed an alias form here
                _append_alias(project, canonical_title=canon_title, alias_title=raw_title, source_step=self.name)

            likely_per = bool(exp.get("likely_per_subject_folder", False))
            per_dir = (exp.get("per_subject_dir") or "").strip()
            pattern_hint = exp.get("filename_pattern_hint") or ""
            sheets_same = bool(exp.get("sheets_expected_same", True))
            decision = (exp.get("decision") or "").strip()

            if decision == "not_trialwise_applicable":
                continue

            for cand in (exp.get("candidates") or []):
                rel = cand.get("relpath") or ""
                if not rel:
                    continue

                # if LLM suggested per-subject directory, use that; else fall back to candidate's parent
                parent = per_dir if per_dir else _parent(rel)
                siblings = by_dir.get(parent, [])
                if not siblings:
                    continue

                cand_sheet = (cand.get("sheet") or "").strip()
                cand_sheet_norm: Optional[str] = cand_sheet if cand_sheet != "" else None

                seed = by_key.get((rel, cand_sheet_norm))
                if not seed:
                    # try without sheet if unknown
                    seed = by_key.get((rel, None))
                    if not seed:
                        continue

                seed_headers = seed.get("columns_head") or []
                seed_skel = _make_skeleton(_basename(rel))

                members: List[Dict[str, Any]] = []
                subj_keys: List[str] = []

                for s in siblings:
                    s_rel = s.get("relpath") or ""
                    s_sheet = s.get("sheet")
                    s_sheet_norm = s_sheet if s_sheet is None else str(s_sheet)

                    # If sheets are expected the same (Excel), enforce it; else allow any
                    if sheets_same and cand_sheet_norm is not None and cand_sheet_norm != s_sheet_norm:
                        continue

                    s_headers = s.get("columns_head") or []
                    j = _headers_jaccard(seed_headers, s_headers)
                    skel_match = (_make_skeleton(_basename(s_rel)) == seed_skel)

                    if skel_match or j >= J_MIN:
                        members.append({
                            "relpath": s_rel,
                            "sheet": s_sheet_norm,
                            "columns_head": s_headers,
                        })
                        sk = _extract_subject_key_from_name(s_rel)
                        if sk:
                            subj_keys.append(sk)

                # acceptance criteria (flexible)
                n_mem = len(members)
                n_subj = len(set(subj_keys))

                accept = False
                if likely_per:
                    # With LLM hint: ≥2 members is enough
                    accept = n_mem >= 2
                else:
                    # Without hint: require stronger evidence
                    accept = (n_mem >= 3 and n_subj >= 2)

                if accept:
                    groups.append({
                        "experiment_title": canon_title,  # canonicalized
                        "decision": decision,
                        "source_candidate": {
                            "relpath": rel,
                            "sheet": cand_sheet_norm,
                            "format": cand.get("format", ""),
                            "score": cand.get("score", 0.0),
                            "reason": cand.get("reason", ""),
                            "subject_id_hints": cand.get("subject_id_hints", []),
                            "trial_index_hints": cand.get("trial_index_hints", []),
                            "condition_hints": cand.get("condition_hints", []),
                            "filters": cand.get("filters", []),
                        },
                        "group": {
                            "dir": parent,
                            "sheet": cand_sheet_norm if sheets_same else None,
                            "pattern_hint": pattern_hint if pattern_hint else seed_skel,
                            "members": members,
                            "n_members": n_mem,
                            "n_distinct_subjects": n_subj,
                            "rules": {
                                "header_jaccard_min": J_MIN,
                                "skeleton_considered": True,
                                "llm_hint_used": likely_per,
                                "sheets_expected_same": sheets_same,
                            }
                        }
                    })
                    total_members += n_mem

        return {
            "groups": groups,
            "summary": {"n_groups": len(groups), "total_members": total_members},
            "notes": [
                "Expansion triggered primarily by LLM hint likely_per_subject_folder=true; otherwise stricter checks apply.",
                "Filename pattern detection is position-agnostic (leading/middle/trailing id-like tokens).",
                "Members accepted if filename skeleton matches OR header Jaccard >= 0.50.",
                "Accepted when LLM hinted True and ≥2 members, else require ≥3 members and ≥2 distinct subject keys.",
                "Experiment titles are canonicalized using DescribeExperiments and experiment_aliases.json; new aliases are appended automatically.",
            ],
        }
