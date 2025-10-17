# sweetExtract/steps/llm_review_sb_program.py
from __future__ import annotations
import json
import os
import re
import ast
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Set

from sweetExtract.project import Project
from sweetExtract.steps.base import BaseStep
from sweetExtract.steps.llm_complete_sb_program_from_draft import LLMCompleteSBProgram

# ---------- tiny IO helpers ----------
def _read_json(p: Path) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _read_text(p: Path) -> str:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _clip(s: str, max_chars: int) -> str:
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + "\n…[truncated]")

# ---------- package docs & examples ----------
def _load_init_docs() -> Dict[str, Any]:
    try:
        p = files("sweetExtract.info.sb").joinpath("init_docs.json")
        return json.loads(p.read_text(encoding="utf-8")) if p else {}
    except Exception:
        return {}

def _load_examples() -> Dict[str, Any]:
    try:
        p = files("sweetExtract.info.sb").joinpath("examples.json")
        return json.loads(p.read_text(encoding="utf-8")) if p else {}
    except Exception:
        return {}

# Reuse the renderers from the prior step if present; otherwise provide safe fallbacks.
try:
    from sweetExtract.steps.llm_complete_sb_program_from_draft import _init_docs_for_stimuli, _examples_for_stimuli
except Exception:
    def _init_docs_for_stimuli(stimuli: List[str], init_docs: Dict[str, Any]) -> str:
        out: List[str] = []
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for _, v in (init_docs or {}).items():
            if isinstance(v, dict):
                cls = v.get("class_name") or ""
                if cls:
                    by_class.setdefault(cls, []).append(v)
        for cls in stimuli:
            for e in by_class.get(cls, []):
                sig = (e.get("signature") or "").strip()
                doc = (e.get("init_docstring") or "").strip()
                out.append(f"### {cls}\n```python\n{sig}\n```\n```text\n{doc}\n```")
        return "\n\n".join(out).strip() or "(no init docs)"
    def _examples_for_stimuli(stimuli: List[str], examples: Dict[str, Any], max_chars: int = 5000) -> str:
        blocks: List[str] = []
        total = 0
        for cls in stimuli:
            key = cls.lower()
            raw = examples.get(key, [])
            ex_list = raw if isinstance(raw, list) else [raw]
            for ex in ex_list:
                code = (ex.get("code") or "").strip() if isinstance(ex, dict) else ""
                if not code: continue
                chunk = f"#### Example: {cls}\n```python\n{code}\n```"
                if total + len(chunk) > max_chars:
                    remain = max_chars - total
                    if remain > 0:
                        blocks.append(chunk[:remain])
                    total = max_chars
                    break
                blocks.append(chunk)
                total += len(chunk)
            if total >= max_chars:
                break
        return "\n\n".join(blocks).strip() or "(no examples)"

# ---------- generic stimulus detection (no hardcoding) ----------
_STIM_IMPORT_RE = re.compile(r"from\s+sweetbean\.stimulus\s+import\s+([^\n]+)")
_CLASS_NAME_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\b")

def _detect_used_stimuli(code: str) -> List[str]:
    """Collect probable class names from imports and callsites (best-effort, generic)."""
    used: Set[str] = set()
    # import lines
    for m in _STIM_IMPORT_RE.finditer(code or ""):
        names = m.group(1)
        for n in names.split(","):
            nm = n.strip().split(" as ")[0].strip()
            if nm and nm[0].isupper():
                used.add(nm)
    # callsites by AST (Name(...))
    try:
        tree = ast.parse(code or "")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id and fn.id[0].isupper():
                    used.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    # something.BilateralRSVP(...)
                    attr = fn.attr
                    if attr and attr[0].isupper():
                        used.add(attr)
    except Exception:
        # fall back: scan CamelCase tokens
        for m in _CLASS_NAME_RE.finditer(code or ""):
            used.add(m.group(1))
    return sorted(used)

# ---------- LLM-only review step ----------
class LLMReviewSBProgram(BaseStep):
    """
    LLM-only conceptual/API review for SweetBean programs.
    Inputs (from prior steps): completed program spec + meta description/docs/examples.
    Output: meta/llm_sb_program_llm_review.json (array of review records).
    """
    artifact_is_list = True

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_review_sb_program",
            artifact="meta/llm_sb_program_llm_review.json",
            depends_on=[LLMCompleteSBProgram],
            map_over=LLMCompleteSBProgram,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # ---------- LLM call ----------
    def _llm_review(self, description: str, code: str, init_docs_txt: str, examples_txt: str) -> Dict[str, Any]:
        """
        Ask the LLM for a structured issue list. No hard rules, no simulation.
        """
        try:
            from sweetExtract.utils.llm_client import generate_response
        except Exception:
            # if client not available, return empty advisories
            return {"issues": [], "verdict": "unknown", "notes": ["LLM client unavailable"]}

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "category": {"type": "string"},
                            "severity": {"type": "string", "enum": ["error", "warn", "note"]},
                            "summary": {"type": "string"}
                        },
                        "required": ["category", "severity", "summary"]
                    }
                },
                "verdict": {"type": "string", "enum": ["pass", "attention", "fail", "unknown"]},
                "notes": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["issues", "verdict", "notes"]
        }

        # Keep things general, non-hardcoded, and review-only.
        system_prompt = (
            "You are a senior experiment-tools reviewer. "
            "Review the provided SweetBean experiment program for conceptual/API problems.\n"
            "Rules:\n"
            " • Use ONLY the information in the description, code, init-docstrings, and examples.\n"
            " • Do NOT invent API fields or behavior. If uncertain, mark severity lower and state uncertainty.\n"
            " • Focus on conceptual issues most. \n"
            "Output strictly as JSON conforming to the provided schema."
        )

        prompt = (
            "=== EXPERIMENT DESCRIPTION ===\n" + _clip(description, 12000) + "\n\n"
            "=== PROGRAM CODE (review for conceptual/API issues) ===\n" + _clip(code, 24000) + "\n\n"
            "=== INIT DOCSTRINGS (relevant stimuli) ===\n" + _clip(init_docs_txt, 18000) + "\n\n"
            "=== EXAMPLES (relevant stimuli) ===\n" + _clip(examples_txt, 12000) + "\n\n"
            "Task: Identify issues and errors."
        )

        out = generate_response(
            model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
            system_prompt=system_prompt,
            prompt=prompt,
            json_schema=schema,
            schema_name="SBProgramLLMReview_v1",
            strict_schema=True,
            reasoning_effort="high",
            text_verbosity="low",
        ) or {}

        # Add a default verdict if missing
        issues = out.get("issues") or []
        if "verdict" not in out:
            has_err = any(i.get("severity") == "error" for i in issues)
            has_warn = any(i.get("severity") == "warn" for i in issues)
            out["verdict"] = "fail" if has_err else ("attention" if has_warn else "pass")
        if "notes" not in out:
            out["notes"] = []
        return out

    # ---------- mapped compute ----------
    def compute_one(self, project: Project, item: Dict[str, Any], idx: int,
                    all_items: List[Dict[str, Any]], prior: List[Dict[str, Any]]) -> Dict[str, Any]:

        title = (item or {}).get("experiment_title") or f"Experiment {idx+1}"
        slug = (item or {}).get("slug") or "exp"
        py_rel = (item or {}).get("python_file") or ""
        code = (project.artifacts_dir / py_rel).read_text(encoding="utf-8") if py_rel else (item.get("python_code") or "")

        # Description (best-effort lookup by title across common containers)
        detailed = _read_json(project.artifacts_dir / "meta" / "experiments_empirical_detailed.json") or {}
        def _desc_for_title(coll: Dict[str, Any], title: str) -> str:
            tnorm = (title or "").strip().lower()
            for it in (coll.get("items") or coll.get("programs") or coll.get("data") or coll.get("results") or []):
                nm = (it.get("title") or it.get("experiment_title") or it.get("slug") or "").strip().lower()
                aliases = [a.strip().lower() for a in (it.get("aliases") or [])]
                if nm == tnorm or tnorm in aliases:
                    return (it.get("standalone_summary") or it.get("description") or it.get("methods_text") or "").strip()
            return ""
        description = _desc_for_title(detailed, title)

        # Detect relevant stimuli, then gather only those docs/examples (keeps prompt compact)
        init_docs = _load_init_docs()
        examples = _load_examples()
        used_stimuli = _detect_used_stimuli(code)
        init_docs_txt = _init_docs_for_stimuli(used_stimuli, init_docs)
        examples_txt  = _examples_for_stimuli(used_stimuli, examples)

        review = self._llm_review(description, code, init_docs_txt, examples_txt)

        return {
            "experiment_title": title,
            "slug": slug,
            "python_file": py_rel,
            "verdict": review.get("verdict", "unknown"),
            "issues": review.get("issues", []),
            "notes": review.get("notes", []),
        }
