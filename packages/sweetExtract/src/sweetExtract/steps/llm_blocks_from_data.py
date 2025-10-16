# sweetExtract/steps/llm_blocks_from_data.py
from __future__ import annotations
import json, os, re, textwrap
from typing import Any, Dict, List, Tuple, Optional

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.sb_trial_schema_for_llm import SBTrialSchemaForLLM
from sweetExtract.steps.llm_blocks_from_description import LLMBlocksFromDescription

# ---------- io helpers ----------

def _read_json(path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_trial_schema_for_llm(project: Project) -> Dict[str, Dict[str, Any]]:
    candidates = [
        project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json",
        project.artifacts_dir / "meta" / "sb_trial_schema_llm.json",
    ]
    for p in candidates:
        obj = _read_json(p)
        if obj and isinstance(obj.get("items"), list):
            out: Dict[str, Dict[str, Any]] = {}
            for it in obj["items"]:
                title = (it or {}).get("experiment_title") or (it or {}).get("title") or ""
                if title:
                    out[title] = it
            if out:
                return out
    return {}

def _load_blocks_from_description(project: Project) -> Dict[str, Dict[str, Any]]:
    """
    Load Step 1 output (conceptual blocks + extracted text).
      artifacts/meta/llm_blocks_from_description.json
      { "items": [ { "experiment_title": "...", "plan": { block_description_text, ... } }, ... ] }
    """
    p = project.artifacts_dir / "meta" / "llm_blocks_from_description.json"
    obj = _read_json(p) or {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj.get("items"), list):
        for it in obj["items"]:
            title = (it or {}).get("experiment_title") or ""
            if title:
                out[title] = it
    return out

# ---------- candidate ranking (soft, not enforced) ----------

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

# Name cues (soft): earlier entries imply stronger “block-likeness”.
_NAME_CUES = [
    "block", "blk", "blockid", "blocklabel",
    "trialblock", "experimentblock", "expblock", "runblock",
    "session", "sess", "run", "phase", "part", "half",
    "segment", "epoch", "round", "series", "cycle", "set",
    "stage", "section", "period", "main", "training", "test",
]

def _name_score(name: str) -> int:
    nk = _norm_key(name)
    score = 1_000  # default: weak
    for idx, cue in enumerate(_NAME_CUES):
        if cue in nk:
            score = min(score, idx)  # lower is better
    return score

def _cardinality_score(n_unique: Optional[int]) -> int:
    """
    Soft preference for small-to-moderate cardinality, but *do not* punish
    typical within-session chunking. Blocks need not be 'conceptual'.
    """
    if not isinstance(n_unique, int) or n_unique <= 0:
        return 200  # unknown gets a mild penalty
    if n_unique == 1:
        return 150  # not useful to split
    # Within-session blocks are often 8–24, but allow broad ranges without punishment.
    if 2 <= n_unique <= 200:
        return 0    # sweet spot for robust splitting—even if many block levels
    if 201 <= n_unique <= 500:
        return 30   # mild penalty: still acceptable
    return 90       # very high: likely trial-level or too granular in practice

def _rank_block_candidates(schema_obj: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    headers: List[str] = list(schema_obj.get("headers") or [])
    cols = schema_obj.get("columns") or {}
    ranked: List[Tuple[str, int, int]] = []  # (name, name_score, card_score)
    for h in headers:
        meta = cols.get(h) or {}
        n_unique = meta.get("n_unique")
        ranked.append((h, _name_score(h), _cardinality_score(n_unique)))
    # Sort by (name_score, cardinality_score, original order)
    ranked.sort(key=lambda x: (x[1], x[2], headers.index(x[0]) if x[0] in headers else 10_000))
    return ranked

# ---------- LLM call (schema enforces exact header tokens) ----------

def _generate_from_data(prompt: str, allowed_headers: List[str]) -> Dict[str, Any]:
    """
    Build a strict JSON schema that makes split_by.column an enum over allowed_headers.
    This prevents punctuation/brace artifacts and forces exact header tokens.
    """
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {}

    # selection schema
    selection_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "where":   {"type": "string"},
            "first_n": {"type": ["integer", "null"]},
            "last_n":  {"type": ["integer", "null"]},
            "sample_n":{"type": ["integer", "null"]},
            "percent": {"type": ["number",  "null"]},
            "column":  {"type": ["string",  "null"]},
            "distinct":{"type": ["boolean", "null"]},
        },
        # strict_schema: include every key here
        "required": ["where","first_n","last_n","sample_n","percent","column","distinct"]
    }

    # block schema
    block_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "criteria": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"where": {"type": "string"}},
                "required": ["where"]
            },
            "order": {"type": ["string","null"]},
            "repeat": {"type": ["integer","null"]},
            "randomize_within": {"type": ["boolean","null"]},
            "selection": selection_schema,
            "gating": {"type": ["string","null"]},
            "trial_order": {"type": ["string","null"]},
            "trial_order_column": {"type": ["string","null"]},
        },
        # strict_schema: include every key here
        "required": ["label","criteria","order","repeat","randomize_within",
                     "selection","gating","trial_order","trial_order_column"]
    }

    # split_by schema — enforce exact header via enum when possible
    column_prop: Dict[str, Any]
    if allowed_headers:
        column_prop = {"type": "string", "enum": allowed_headers}
    else:
        column_prop = {"type": "string"}  # degenerate case; no headers surfaced

    split_by_schema = {
        "type": ["object","null"],
        "additionalProperties": False,
        "properties": {
            "column": column_prop,
            # Optional explicit values; keep nullable. Consumer usually infers uniques.
            "values": { "type": ["array","null"], "items": {"type": "string"} }
        },
        # strict_schema: include every key from properties (values can be null)
        "required": ["column","values"]
    }

    # top-level schema
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "split_by": split_by_schema,                      # preferred when a single column clearly encodes blocks
            "blocks": { "type": "array", "items": block_schema },  # fallback for exotic cases
            "block_order_policy": {"type": "string"},
            "block_order_column": {"type": ["string","null"]},
            "counterbalance": { "type": ["array","null"], "items": {"type": "string"} },
            "notes": { "type": ["array","null"], "items": {"type": "string"} }
        },
        # strict_schema: require all keys; split_by may be null
        "required": ["split_by","blocks","block_order_policy","block_order_column","counterbalance","notes"]
    }

    system_prompt = (
        "Derive a BLOCK PLAN from the dataset schema. Use provided column names EXACTLY (case-sensitive) and prefer what the data support.\n\n"
        "CLARIFICATION ABOUT BLOCKS:\n"
        "• Blocks do NOT need conceptual differences. In many experiments (e.g., psychophysics/neuro/RSVP), blocks are simply within-session, "
        "  time-ordered chunks of trials. That is valid and often the desired partition.\n"
        "• Therefore, if a column appears to encode trial blocks (e.g., 'Block', 'trial_block', etc.), prefer splitting by that column—even if it has many levels.\n\n"
        "SOFT PRIORITY POLICY (guidance, not hard rules):\n"
        "• First, look for a single column whose NAME suggests block-like units (case-insensitive cues: block/blk, block_id/label, trial_block, "
        "  experiment_block/exp_block, run_block, session/sess, run, phase, part/half, segment, epoch, round, series, cycle, set, stage, section, period, training/test/main).\n"
        "• Prefer small-to-moderate cardinality, but do not avoid a reasonable block column merely because it has multiple levels; within-session chunking is expected.\n"
        "• When a single suitable column exists, use the concise representation: split_by={\"column\":\"<ExactHeaderName>\"} "
        "(values are treated as opaque strings like '1', 'A', '1a').\n"
        "• If no single clear column exists, you may specify explicit blocks with simple 'where' expressions, or return a single block with where='true'.\n"
        "• Keep outputs minimal and readable; do not invent columns. If you pick a less-obvious column over a more obvious one, include a brief rationale in 'notes'.\n\n"
        "OUTPUT:\n"
        "• Return JSON that matches the schema exactly.\n"
        "• Always include the key 'split_by' (set to null if not used).\n"
        "• Prefer top-level 'split_by' when appropriate; when using 'split_by', you may leave 'blocks' as [].\n"
        "• Avoid enumerating 'values' for 'split_by' unless truly necessary (the consumer can infer uniques)."
    )

    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=system_prompt,
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMBlocksFromData",
        strict_schema=True,
        reasoning_effort="medium",
        text_verbosity="low",
    ) or {}

# ---------- prompt builders ----------

def _mk_column_section(schema_obj: Dict[str, Any], max_samples: int = 10) -> str:
    headers = list(schema_obj.get("headers") or [])
    cols = schema_obj.get("columns") or {}
    lines = ["COLUMNS:"]
    for name in headers:
        samples = []
        truncated = None
        n_unique = None
        meta = cols.get(name) or {}
        if isinstance(meta.get("samples"), list):
            samples = meta["samples"][:max_samples]
        if isinstance(meta.get("sample_truncated"), bool):
            truncated = meta["sample_truncated"]
        if isinstance(meta.get("n_unique"), int):
            n_unique = meta["n_unique"]
        lines.append(
            f"- {name}:\n"
            f"    samples={samples}\n"
            f"    sample_truncated={truncated}\n"
            f"    n_unique={n_unique}"
        )
    if not headers:
        lines.append("(no columns)")
    return "\n".join(lines)

def _mk_description_hint(desc_item: Dict[str, Any]) -> str:
    plan = (desc_item or {}).get("plan") or {}
    txt = plan.get("block_description_text") or ""
    labels = [ (b or {}).get("label") for b in (plan.get("blocks") or []) if isinstance(b, dict) ]
    labels = [l for l in labels if isinstance(l, str) and l]
    hint = "BLOCK DESCRIPTION (from paper; hint only):\n" + (txt.strip() or "(none)")
    if labels:
        hint += f"\nCONCEPTUAL BLOCK LABELS: {labels}"
    if isinstance(plan.get("block_count"), int):
        hint += f"\nCONCEPTUAL BLOCK COUNT: {plan['block_count']}"
    return hint

def _mk_priority_hints(schema_obj: Dict[str, Any]) -> str:
    ranked = _rank_block_candidates(schema_obj)
    cols = schema_obj.get("columns") or {}
    top = ranked[:8]
    parts = []
    for name, ns, cs in top:
        n_unique = (cols.get(name) or {}).get("n_unique")
        parts.append(f"{name} (name_score={ns}, n_unique={n_unique}, card_score={cs})")
    lines = ["PRIORITY HINTS (derived from column names + cardinality):"]
    lines.append("- Candidate block columns (best-first): " + (", ".join(parts) if parts else "(none)"))
    lines.append("- Reminder: blocks can be simple within-session chunks; high-ish n_unique is acceptable if it matches 'Block'-like intent.")
    return "\n".join(lines)

# ---------- the step ----------

class LLMBlocksFromData(BaseStep):
    """
    Step 2/4: infer a concrete (data-backed) block plan from the dataset schema,
    using the Step 1 textual block description as a hint (not ground truth).

    Inputs:
      • artifacts/meta/sb_trial_schema_for_llm.json  (or ..._llm.json legacy)
      • artifacts/meta/llm_blocks_from_description.json  (Step 1)
      • DescribeExperiments (only for titles to map over)

    Output (combined):
      • artifacts/meta/llm_blocks_from_data.json
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_blocks_from_data",
            artifact="meta/llm_blocks_from_data.json",
            depends_on=[FilterEmpiricalExperiments, SBTrialSchemaForLLM, LLMBlocksFromDescription],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        dst = project.artifacts_dir / self.artifact
        return True if self._force else not dst.exists()

    # mapped; BaseStep aggregates per-item results
    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or item.get("experiment_title") or f"Experiment {idx+1}"

        schema_map = _load_trial_schema_for_llm(project)
        desc_map = _load_blocks_from_description(project)

        if title not in schema_map:
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_schema_for_title",
            }

        schema_obj = schema_map.get(title) or {}
        desc_item = desc_map.get(title) or {}

        headers = list(schema_obj.get("headers") or [])
        col_section = _mk_column_section(schema_obj)
        hint = _mk_description_hint(desc_item)
        priority_hints = _mk_priority_hints(schema_obj)

        allowed_headers_literal = "[" + ", ".join(repr(h) for h in headers) + "]"

        prompt = textwrap.dedent(f"""
        TITLE:
        {title}
        
        Given the following information about trial-wise dataset for a single participant in an experiment,
        your goal is to figure out if and how the trials where divided into BLOCKS (e.g., within-session chunks).

        DATASET SCHEMA (authoritative; use *only* these columns in expressions):
        {col_section}

        {priority_hints}

        {hint}

        ALLOWED_HEADERS (exact token choices for split_by.column):
        {allowed_headers_literal}

        TASK:
        Propose a concrete BLOCK PLAN. Follow the guidance:
        • Blocks need not be conceptually distinct; within-session trial chunks are acceptable and preferred when encoded (e.g., by a 'Block'-like column).
        • Prefer a single clear block column and use split_by={{"column":"<ExactHeaderName>"}} when appropriate.
        • IMPORTANT: split_by.column MUST be exactly one token from ALLOWED_HEADERS. Do not add punctuation or braces; do not write an expression.
        • Treat values as opaque strings; do not coerce.
        • If no single clear column exists, you may specify explicit blocks with simple 'where' or return a single block (where='true').
        • Keep things minimal and robust across datasets. If you pick a less-obvious column over an obvious one, add a brief rationale in 'notes'.
        """)

        raw = _generate_from_data(prompt, headers) or {}

        # Normalize presence/absence
        blocks = list(raw.get("blocks") or [])
        split_by = raw.get("split_by") if isinstance(raw.get("split_by"), dict) else None

        # Prefer split_by when present to keep output unambiguous and column-driven
        if isinstance(split_by, dict) and isinstance(split_by.get("column"), str):
            blocks = []

        # Final fallback: if neither is present, emit a single trivial block.
        if not split_by and not blocks:
            blocks = [{
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
            }]
            raw["block_order_policy"] = raw.get("block_order_policy") or "as_listed"
            raw["block_order_column"] = raw.get("block_order_column") or None
            raw["counterbalance"] = raw.get("counterbalance") or None
            base_notes = raw.get("notes") or []
            if base_notes is None:
                base_notes = []
            base_notes = list(base_notes)
            base_notes.append("Fallback: no clear single-column block indicator detected.")
            raw["notes"] = base_notes

        # diagnostics: which columns were referenced (for transparency)
        headers_list = headers
        used: List[str] = []

        def _collect_used(expr: str):
            if not isinstance(expr, str):
                return
            for h in headers_list:
                if h and (h in expr) and (h not in used):
                    used.append(h)

        if blocks:
            for bl in blocks:
                _collect_used((bl.get("criteria") or {}).get("where", ""))
                _collect_used((bl.get("selection") or {}).get("where", ""))

        if isinstance(split_by, dict) and isinstance(split_by.get("column"), str):
            col = split_by["column"]
            if col not in used:
                used.append(col)

        out = {
            "experiment_title": title,
            "status": "ok",
            "block_plan": {
                "split_by": split_by or None,
                "blocks": blocks,
                "block_order_policy": raw.get("block_order_policy") or "as_listed",
                "block_order_column": raw.get("block_order_column"),
                "counterbalance": raw.get("counterbalance"),
                "notes": raw.get("notes"),
            },
            "diagnostics": {
                "allowed_columns": headers_list,
                "used_columns": used,
            },
        }
        return out
