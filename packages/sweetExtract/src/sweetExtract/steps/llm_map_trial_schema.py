# sweetExtract/steps/llm_map_trial_schema.py
from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Optional, Tuple

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.describe_experiments import DescribeExperiments

# ---------- helpers: load & prompt snippets ----------

def _load_schema_item(artifacts_dir, idx: int, title: str) -> Optional[Dict[str, Any]]:
    """
    Load one item from meta/sb_trial_schema_for_llm.json for THIS experiment.
    Prefers map_over index alignment; falls back to exact title match.
    """
    p = artifacts_dir / "meta" / "sb_trial_schema_for_llm.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    items = (obj.get("items") or []) if isinstance(obj, dict) else []
    if 0 <= idx < len(items) and (items[idx] or {}).get("status") == "ok":
        return items[idx]
    for it in items:
        if (it or {}).get("status") == "ok" and (it or {}).get("experiment_title") == title:
            return it
    return None


def _clip_schema_for_prompt(prev: Dict[str, Any], max_cols: int = 80, max_vals: int = 10) -> str:
    """
    Turn one sb_trial_schema_for_llm item into a compact, LLM-friendly text block.
    """
    title = prev.get("experiment_title", "")
    headers = prev.get("headers") or []
    cols = prev.get("columns") or {}
    # keep order by headers, then extras
    ordered = list(headers) + [c for c in cols.keys() if c not in headers]
    lines = [f"EXPERIMENT: {title}", f"HEADERS ({len(headers)}): {', '.join(headers[:max_cols])}"]
    lines.append("COLUMNS (unique samples):")
    for c in ordered[:max_cols]:
        ex = cols.get(c, {}).get("samples") or []
        trunc = cols.get(c, {}).get("sample_truncated", False)
        exs = ", ".join(str(x) for x in ex[:max_vals])
        tag = " (sample)" if trunc else ""
        lines.append(f"- {c}: [{exs}]{tag}")
    return "\n".join(lines)


def _generate(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        # fallback: schema-shaped empty
        return {
            "experiment_title": "",
            "id_fields": [],
            "presentation": [],
            "response": [],
            "unused_columns": [],
            "diagnostics": {"notes": ["LLM unavailable"], "confidence": 0.0, "model": ""}
        }

    out = generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You are mapping one experiment's trial-wise data for a general analysis engine.\n"
            "You will receive (1) a standalone experiment description and (2) a data description consisting of exact headers and sampled values.\n"
            "Your task is to select and group columns needed for:\n"
            "  A) stimulus/presentation (anything shown/played/displayed/asked/procedure parameters/options, etc.)\n"
            "  B) response (participant/system responses: keys/clicks/choices/free text/scores/Rt/physio/etc.)\n"
            "Also identify ID fields needed to uniquely index rows (e.g., subject/session/block/trial), **but use only headers that actually exist**.\n\n"
            "Rules:\n"
            "- Choose ONLY among the listed headers; do not invent, rename, normalize, or derive new columns.\n"
            "- Group presentation and response columns into meaningful groups with short names and a brief justification.\n"
            "- If some headers are irrelevant for presentation/response or IDs, list them under unused_columns.\n"
            "- Be minimal but complete: include what's needed to reconstruct stimuli/procedure and responses for this experiment.\n"
            "Return ONLY JSON that matches the schema."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLMExperimentIOMap",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return out or {}

# ---------- validation & cleanup ----------

def _order_like(headers: List[str], subset: List[str]) -> List[str]:
    H = [h for h in subset if h in headers]
    seen = set()
    out = []
    for h in headers:
        if h in H and h not in seen:
            out.append(h); seen.add(h)
    return out

def _clean_group(headers: List[str], grp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(grp, dict):
        return None
    name = str(grp.get("name") or "").strip()
    cols = [c for c in (grp.get("columns") or []) if isinstance(c, str)]
    just = str(grp.get("justification") or "").strip()
    cols = _order_like(headers, list(dict.fromkeys(cols)))
    if not name or not cols:
        return None
    return {"name": name, "columns": cols, "justification": just}

def _validate_and_fix(
    headers: List[str],
    id_fields: List[str],
    presentation: List[Dict[str, Any]],
    response: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """
    - keep only existing headers; dedupe; preserve order
    - drop empty groups
    - compute unused_columns
    - notes for anything dropped
    """
    notes: List[str] = []

    id_fields = _order_like(headers, list(dict.fromkeys([c for c in id_fields if isinstance(c, str)])))
    pres = []
    for g in (presentation or []):
        cg = _clean_group(headers, g)
        if cg: pres.append(cg)
        else: notes.append("dropped_empty_or_unknown_presentation_group")

    resp = []
    for g in (response or []):
        cg = _clean_group(headers, g)
        if cg: resp.append(cg)
        else: notes.append("dropped_empty_or_unknown_response_group")

    used = set(id_fields)
    for g in pres: used.update(g["columns"])
    for g in resp: used.update(g["columns"])
    unused = [h for h in headers if h not in used]

    return id_fields, pres, resp, unused, notes

# ---------- step ----------

class LLMMapTrialSchema(BaseStep):
    """
    GENERAL per-experiment I/O mapping (no hardcoded DV names).

    Inputs:
      - DescribeExperiments (per-experiment description)
      - meta/sb_trial_schema_for_llm.json  (headers + sampled values)

    Outputs:
      - meta/llm_experiment_io_map.json (combined)
      - artifacts/llm_experiment_io_map/{idx}.json (per-experiment)
    """

    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False, max_cols_in_prompt: int = 80, max_vals_per_col: int = 10):
        super().__init__(
            name="llm_map_trial_schema",
            artifact="meta/llm_experiment_io_map.json",
            depends_on=[DescribeExperiments],
            map_over=DescribeExperiments,
        )
        self._force = bool(force)
        self.max_cols_in_prompt = max_cols_in_prompt
        self.max_vals_per_col   = max_vals_per_col

    def should_run(self, project: Project) -> bool:
        src = project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json"
        dst = project.artifacts_dir / self.artifact
        if not src.exists():
            return False
        if self._force:
            return True
        return not dst.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        raise NotImplementedError  # mapped step; BaseStep aggregates per-item outputs

    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = item.get("title") or f"Experiment {idx+1}"
        desc  = (item.get("standalone_summary") or "").strip()

        prev = _load_schema_item(project.artifacts_dir, idx, title)
        if not prev or prev.get("status") != "ok":
            return {
                "experiment_title": title,
                "status": "skipped",
                "reason": "no_trial_schema_for_llm",
                "id_fields": [],
                "presentation": [],
                "response": [],
                "unused_columns": [],
                "diagnostics": {"notes": []}
            }

        snippet = _clip_schema_for_prompt(prev, self.max_cols_in_prompt, self.max_vals_per_col)
        headers = prev.get("headers") or []

        # JSON schema the LLM must follow
        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experiment_title": {"type": "string"},
                "id_fields": {"type": "array", "items": {"type": "string"}},
                "presentation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "columns": {"type": "array", "items": {"type": "string"}},
                            "justification": {"type": "string"}
                        },
                        "required": ["name","columns","justification"]
                    }
                },
                "response": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "columns": {"type": "array", "items": {"type": "string"}},
                            "justification": {"type": "string"}
                        },
                        "required": ["name","columns","justification"]
                    }
                },
                "unused_columns": {"type": "array", "items": {"type": "string"}},
                "diagnostics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "notes": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                        "model": {"type": "string"}
                    },
                    "required": ["notes","confidence","model"]
                }
            },
            "required": ["experiment_title","id_fields","presentation","response","unused_columns","diagnostics"]
        }

        prompt = (
            f"TARGET EXPERIMENT: {title}\n\n"
            f"EXPERIMENT DESCRIPTION (standalone):\n{desc}\n\n"
            "DATA DESCRIPTION (headers + unique value samples):\n"
            f"{snippet}\n\n"
            "TASK:\n"
            "- Choose ID fields needed to uniquely identify each row (use only existing headers).\n"
            "- Group PRESENTATION columns (stimuli/procedure/options/parameters/etc.) into a few named groups.\n"
            "- Group RESPONSE columns (keys/clicks/choices/text/scores/RT/physio/etc.) into a few named groups.\n"
            "- Put everything not needed in unused_columns.\n"
            "- Use ONLY the provided headers; do not invent or rename."
        )

        out = _generate(prompt, schema)
        out.setdefault("experiment_title", title)

        # Validate & clean
        id_fields = out.get("id_fields") or []
        presentation = out.get("presentation") or []
        response = out.get("response") or []
        id_fixed, pres_fixed, resp_fixed, unused, vnotes = _validate_and_fix(headers, id_fields, presentation, response)

        diag = out.get("diagnostics") or {"notes": [], "confidence": 0.0, "model": ""}
        diag["notes"] = list((diag.get("notes") or [])) + vnotes

        return {
            "experiment_title": out.get("experiment_title", title),
            "status": "ok",
            "id_fields": id_fixed,
            "presentation": pres_fixed,
            "response": resp_fixed,
            "unused_columns": unused,
            "diagnostics": diag,
        }
