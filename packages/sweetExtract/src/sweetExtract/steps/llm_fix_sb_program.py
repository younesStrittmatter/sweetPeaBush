# sweetExtract/steps/llm_fix_sb_program.py
from __future__ import annotations
import ast
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.project import Project
from sweetExtract.steps.base import BaseStep
from sweetExtract.steps.llm_review_sb_program import LLMReviewSBProgram

# ---------- tiny IO ----------
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

def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n…[truncated]"

def _slug(s: str) -> str:
    import re
    s = re.sub(r"\s+", "-", str(s or "").strip())
    s = re.sub(r"[^a-zA-Z0-9\-_.]", "", s)
    return (s.lower() or "exp")[:120]

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

# Reuse string renderers from your “complete” step if available; else minimal fallbacks.
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
        used = set()
        total = 0
        for cls in stimuli:
            key = cls.lower()
            if key in used: continue
            used.add(key)
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

# ---------- simple stimulus detection ----------
import re as _re
_STIM_IMPORT_RE = _re.compile(r"from\s+sweetbean\.stimulus\s+import\s+([^\n]+)")
def _detect_used_stimuli(code: str) -> List[str]:
    used = []
    seen = set()
    for m in _STIM_IMPORT_RE.finditer(code or ""):
        names = m.group(1)
        for n in names.split(","):
            nm = n.strip().split(" as ")[0].strip()
            if nm and nm[0].isupper() and nm not in seen:
                used.append(nm); seen.add(nm)
    return used

# ---------- LLM fixer step ----------
class LLMFixSBProgram(BaseStep):
    """
    LLM-guided auto-fixer for SweetBean programs.
    Consumes the LLM review record and outputs a patched .py file.
    """
    artifact_is_list = True

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_fix_sb_program",
            artifact="meta/llm_sb_program_fixed.json",
            depends_on=[LLMReviewSBProgram],   # run after review
            map_over=LLMReviewSBProgram,       # one fix per reviewed program
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # Strict, parser-compatible schema (must list all keys; no unions/regex/extra keys)
    _SCHEMA: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "python_code": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["python_code", "notes"]
    }

    def _llm_fix(self, description: str, code: str, issues: List[Dict[str, Any]],
                 init_docs_txt: str, examples_txt: str) -> Dict[str, Any]:
        try:
            from sweetExtract.utils.llm_client import generate_response
        except Exception:
            # No LLM client — return original code unchanged.
            return {"python_code": code, "notes": ["LLM client unavailable; left code unchanged."]}

        system_prompt = (
            "You are an expert SweetBean code mechanic.\n"
            "Fix the program to address the listed issues while preserving the file's structure.\n"
            "Rules:\n"
            " • Keep imports and CSV→timeline loader usage intact.\n"
            " • Only modify code needed to resolve issues; do not add unrelated features.\n"
            " • Conform strictly to the INIT DOCSTRINGS and EXAMPLES provided.\n"
            " • Prefer minimal, local edits over rewrites.\n"
            "Output JSON with exactly: python_code (string) and notes (array of strings).\n"
        )

        prompt = (
            "=== EXPERIMENT DESCRIPTION ===\n" + _clip(description, 12000) + "\n\n"
            "=== CURRENT PROGRAM CODE ===\n" + _clip(code, 24000) + "\n\n"
            "=== ISSUES TO FIX (category | severity | summary) ===\n" +
            "\n".join(f"- {i.get('category','?')} | {i.get('severity','?')} | {i.get('summary','')}" for i in (issues or [])) + "\n\n"
            "=== INIT DOCSTRINGS (relevant stimuli) ===\n" + _clip(init_docs_txt, 18000) + "\n\n"
            "=== EXAMPLES (relevant stimuli) ===\n" + _clip(examples_txt, 12000) + "\n\n"
            "Task: Produce a patched file that resolves the issues above without changing file layout more than necessary."
        )

        out = generate_response(
            model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
            system_prompt=system_prompt,
            prompt=prompt,
            json_schema=self._SCHEMA,
            schema_name="SBProgramLLMFix_v1",
            strict_schema=True,
            reasoning_effort="low",
            text_verbosity="low",
        ) or {}

        # Ensure required keys exist (defensive)
        if not isinstance(out.get("python_code"), str):
            out["python_code"] = code
            notes = out.get("notes") or []
            out["notes"] = notes + ["Model did not return valid python_code; kept original."]
        if not isinstance(out.get("notes"), list):
            out["notes"] = []
        return out

    def compute_one(self, project: Project, item: Dict[str, Any], idx: int,
                    all_items: List[Dict[str, Any]], prior: List[Dict[str, Any]]) -> Dict[str, Any]:

        title = (item or {}).get("experiment_title") or f"Experiment {idx+1}"
        slug = (item or {}).get("slug") or _slug(title)
        py_rel = (item or {}).get("python_file") or ""
        verdict = (item or {}).get("verdict") or "unknown"
        issues = (item or {}).get("issues") or []

        # Load current code
        code = (project.artifacts_dir / py_rel).read_text(encoding="utf-8") if py_rel else ""

        # Load description for context
        detailed = _read_json(project.artifacts_dir / "meta" / "experiments_empirical_detailed.json") or {}
        def _desc_for_title(coll: Dict[str, Any], t: str) -> str:
            tnorm = (t or "").strip().lower()
            for it in (coll.get("items") or coll.get("programs") or coll.get("data") or coll.get("results") or []):
                nm = (it.get("title") or it.get("experiment_title") or it.get("slug") or "").strip().lower()
                aliases = [a.strip().lower() for a in (it.get("aliases") or [])]
                if nm == tnorm or tnorm in aliases:
                    return (it.get("standalone_summary") or it.get("description") or it.get("methods_text") or "").strip()
            return ""
        description = _desc_for_title(detailed, title)

        # Restrict docs/examples to used stimuli to keep prompt compact
        init_docs = _load_init_docs()
        examples = _load_examples()
        used_stimuli = _detect_used_stimuli(code)
        init_docs_txt = _init_docs_for_stimuli(used_stimuli, init_docs)
        examples_txt  = _examples_for_stimuli(used_stimuli, examples)

        # If there are no issues (verdict pass) you can skip; but we still allow “polish” on request.
        fix_out = self._llm_fix(description, code, issues, init_docs_txt, examples_txt)
        new_code = (fix_out.get("python_code") or "").rstrip() + "\n"
        notes = fix_out.get("notes") or []

        # Syntax sanity: only write if it parses; otherwise keep original
        good = True
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            good = False
            notes = notes + [f"SyntaxError after fix: {e}"]
            new_code = code  # revert

        # Write file (suffix _fixed.py)
        out_dir = project.artifacts_dir / "sweetbean" / "programs"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(py_rel).name if py_rel else f"{_slug(title)}.py"
        stem = Path(base).stem
        fixed_name = f"{stem}_fixed.py"
        fixed_path = out_dir / fixed_name
        _write_text(fixed_path, new_code)

        return {
            "experiment_title": title,
            "slug": slug,
            "source_python_file": py_rel,
            "python_file": str(fixed_path.relative_to(project.artifacts_dir)),
            "verdict_before": verdict,
            "issues_fixed": issues,
            "notes": notes,
        }
