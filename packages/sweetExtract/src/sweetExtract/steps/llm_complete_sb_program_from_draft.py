# sweetExtract/steps/llm_complete_sb_program.py
from __future__ import annotations

import json
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Type, Union, Optional, Tuple

from sweetExtract.project import Project
from sweetExtract.steps.base import BaseStep

# Upstream steps (the draft builder is essential here)
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.build_sb_program_draft import BuildSBProgramDraft
from sweetExtract.steps.llm_build_param_functions import LLMBuildParamFunctionsPositional
from sweetExtract.steps.llm_refine_sb_parameter_plan import LLMRefineSBParameterPlan
from sweetExtract.steps.llm_select_timeline_variables import LLMSelectTimelineVariables


# ---------------- IO ----------------
def _read_json(p: Path) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _slug(s: str) -> str:
    s = re.sub(r"\s+", "-", str(s or "").strip())
    s = re.sub(r"[^a-zA-Z0-9\-_.]", "", s)
    return s.lower()[:120] or "exp"


# ------------- package docs & examples -------------
def _load_pkg_init_docs() -> Dict[str, Any]:
    """Load sweetExtract.info.sb/init_docs.json (package resource)."""
    try:
        p = files("sweetExtract.info.sb").joinpath("init_docs.json")
        return json.loads(p.read_text(encoding="utf-8")) if p else {}
    except Exception:
        return {}


def _load_pkg_examples() -> Dict[str, Any]:
    """Load sweetExtract.info.sb/examples.json (package resource)."""
    try:
        p = files("sweetExtract.info.sb").joinpath("examples.json")
        return json.loads(p.read_text(encoding="utf-8")) if p else {}
    except Exception:
        return {}


# Map CamelCase class names -> examples.json keys (lowercase)
_EXAMPLE_KEY_ALIASES = {
    "HtmlKeyboardResponse": "htmlkeyboardresponse",
    "MultiChoiceSurvey": "multichoicesurvey",
    "LikertSurvey": "likertsurvey",
    "RandomObjectKinematogram": "randomobjectkinematogram",
    "RandomDotPatterns": "randomdotpatterns",
    "HtmlChoice": "htmlchoice",
}


def _examples_for_stimuli(stimuli: List[str], examples: Dict[str, Any], max_chars: int = 5000) -> str:
    """
    Return nicely formatted code blocks for only the used stimuli.
    """
    ordered: List[tuple[str, str]] = []
    seen_keys = set()
    for cls in stimuli:
        key = _EXAMPLE_KEY_ALIASES.get(cls, cls).lower()
        if key not in seen_keys:
            ordered.append((cls, key))
            seen_keys.add(key)

    blocks: List[str] = []
    total = 0

    for display_name, key in ordered:
        raw = examples.get(key, [])
        ex_list = raw if isinstance(raw, list) else [raw]

        for ex in ex_list:
            if not isinstance(ex, dict):
                continue
            code = (ex.get("code") or "").strip()
            if not code:
                continue

            chunk = f"#### Example: {display_name}\n```python\n{code}\n```"
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    blocks.append(chunk[:remaining])
                    total = max_chars
                break

            blocks.append(chunk)
            total += len(chunk)

        if total >= max_chars:
            break

    return "\n\n".join(blocks).strip() or "(no examples available for these classes)"


def _init_docs_for_stimuli(stimuli: List[str], init_docs: Dict[str, Any]) -> str:
    """Format signatures + docstrings only for used classes from init_docs.json."""
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for _, v in (init_docs or {}).items():
        if isinstance(v, dict):
            cls = v.get("class_name") or ""
            if cls:
                by_class.setdefault(cls, []).append(v)

    parts: List[str] = []
    for cls in stimuli:
        entries = by_class.get(cls, [])
        if not entries:
            continue
        for e in entries:
            sig = (e.get("signature") or "").strip()
            doc = (e.get("init_docstring") or "").strip()
            parts.append(
                f"### {cls}\nSignature:\n```python\n{sig}\n```\nDocstring:\n```text\n{doc}\n```"
            )
    return "\n\n".join(parts).strip() or "(no init docs found)"


# ---------------- helpers ----------------
def _iter_collection(coll: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Yield items from common container keys: items/programs/data/results."""
    for key in ("items", "programs", "data", "results"):
        seq = coll.get(key)
        if isinstance(seq, list):
            return seq
    return []


def _by_title(coll: Dict[str, Any], title: str) -> Dict[str, Any]:
    """Robust lookup of an entry by title across different container keys."""
    tnorm = (title or "").strip().lower()
    for it in _iter_collection(coll):
        nm = (it.get("experiment_title") or it.get("title") or it.get("slug") or "").strip().lower()
        if nm == tnorm:
            return it
    return {}


def _desc_for_title(detailed: Dict[str, Any], title: str) -> str:
    tnorm = (title or "").strip().lower()
    for it in (detailed.get("items") or []):
        nm = (it.get("title") or "").strip().lower()
        aliases = [a.strip().lower() for a in (it.get("aliases") or [])]
        if nm == tnorm or tnorm in aliases:
            return (it.get("standalone_summary") or it.get("description") or it.get("methods_text") or "").strip()
    return ""


# --------------- LLM ---------------
def _generate_json(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        # If the client isn't available, return empty result (step remains no-op).
        return {"programs": []}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL", "gpt-5"),
        system_prompt=(
            "You will FIX & COMPLETE a SweetBean experiment program draft.\n"
            "Keep this structure:\n"
            "  def get_experiment(path):\n"
            "      # the draft already imports everything and calls a CSV→timeline loader\n"
            "      # keep imports as-is and call the same loader the draft calls\n"
            "      # define columns (exactly the provided list) and build timeline\n"
            "      # paste provided mapping functions (verbatim)\n"
            "      # bind per-trial params via FunctionVariable(name, fct=<fn>, args=[...])\n"
            "      # instantiate stimuli in the given order\n"
            "      # return Experiment([Block([...], timeline=timeline)])\n"
            "\n"
            "CONSTRAINTS (ALWAYS):\n"
            "  • Do NOT modify the import section; assume imports in the draft are correct.\n"
            "  • Do NOT implement, rename, or alter any CSV→timeline loader; call the utility\n"
            "    exactly as the draft already does (e.g., timeline_from_csv or similar).\n"
            "  • Use ONLY constructor parameters that appear in the INIT DOCS for each stimulus.\n"
            "  • Paste the provided mapping function code verbatim; DO NOT rename those functions.\n"
            "    Bind them via FunctionVariable(name='<param>', fct=<fn>, args=[<CSV headers in order>]).\n"
            "  • Do NOT invent columns; use exactly the provided timeline columns.\n"
            "    If the draft defines a COLUMNS constant, set that; otherwise define a local `columns` list.\n"
            "  • Convert fixed_params to proper Python types (e.g., '50'→50, 'false'→False, '4px'→'4px').\n"
            "  • Remove extraneous scaffolds that are unnecessary given the provided mapping functions\n"
            "    (e.g., ad-hoc mapping engines or MAPPING_* dicts present in the draft).\n"
            "  • Fill all '...' and TODOs; leave nothing incomplete. No CLI, prints, or network calls.\n"
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_Complete_SB_Program_V1",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {"programs": []}


class LLMCompleteSBProgram(BaseStep):
    """
    Takes the DRAFT program + positional param functions + refined mapping plan
    and emits a complete, runnable SweetBean Python program per experiment.

    Inputs (must exist):
      - meta/sb_program_drafts.json                 (from BuildSBProgramDraft)
      - sweetbean/programs/draft_<slug>.py          (from BuildSBProgramDraft)
      - sweetbean/programs/draft_<slug>.json        (from BuildSBProgramDraft)
      - meta/llm_param_functions.json               (from LLMBuildParamFunctionsPositional)
      - meta/llm_map_sb_parameters_refined.json     (from LLMRefineSBParameterPlan)
      - meta/llm_timeline_variables.json            (from LLMSelectTimelineVariables)
      - meta/experiments_empirical_detailed.json
      - meta/sb_trial_schema_for_llm.json
      - sweetExtract.info.sb/init_docs.json         (package resource)
      - sweetExtract.info.sb/examples.json          (package resource)

    Output:
      - meta/llm_sb_program_spec.json  (ARRAY of Program objects; since artifact_is_list=True)
        [
          { "experiment_title", "slug", "file_name", "python_file", "python_code", "notes": [...] },
          ...
        ]
      - sweetbean/programs/<file_name> (written from python_code during compute_one)
    """
    artifact_is_list = True  # combined artifact will be an array of Program dicts
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="llm_complete_sb_program",
            artifact="meta/llm_sb_program_spec.json",
            depends_on=[
                BuildSBProgramDraft,
                LLMBuildParamFunctionsPositional,
                LLMRefineSBParameterPlan,
                LLMSelectTimelineVariables,
                FilterEmpiricalExperiments,
            ],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # ----- loaders -----
    def _drafts_index(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "sb_program_drafts.json") or {"programs": []}

    def _functions(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_param_functions.json") or {}

    def _refined_plan(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_map_sb_parameters_refined.json") or {}

    def _timeline_vars(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_timeline_variables.json") or {}

    def _detailed(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "experiments_empirical_detailed.json") or {}

    def _trial_schema(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json") or {}

    # compute (mapped)
    def compute_one(
        self,
        project: Project,
        item: Dict[str, Any],
        idx: int,
        all_items: List[Dict[str, Any]],
        prior: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug = _slug(title)
        file_name_default = f"{slug}.py"

        # Index entries
        drafts_idx = self._drafts_index(project)
        draft_entry = _by_title(drafts_idx, title)

        # Load draft python code
        draft_py_rel = draft_entry.get("python_file") or ""
        draft_py_abs = (project.artifacts_dir / draft_py_rel).resolve()
        draft_code = _read_text(draft_py_abs)

        # Mapping functions (code + args) and binding hints
        func_all = self._functions(project)
        funcs_entry = _by_title(func_all, title)
        func_list = funcs_entry.get("functions") or []
        fn_blocks: List[str] = []
        for f in (func_list or []):
            code = (f or {}).get("python_code") or ""
            if code.strip():
                fn_blocks.append(code.strip())
        functions_text = "\n\n".join(fn_blocks) or "# (no per-trial mapping functions needed)"
        fn_bindings = [
            {
                "param": f.get("param"),
                "fn_name": f.get("fn_name"),
                "args": f.get("args") or [],
                "stimulus": f.get("stimulus"),
            }
            for f in (func_list or [])
        ]

        # Refined mapping plan (units order, fixed_params, per_trial)
        plan_all = self._refined_plan(project)
        plan_entry = _by_title(plan_all, title)
        units = plan_entry.get("units") or []
        stimuli_order = [u.get("stimulus") or u.get("class_name") for u in units if (u.get("stimulus") or u.get("class_name"))]

        # Timeline columns (selected)
        tl_all = self._timeline_vars(project)
        tl_entry = _by_title(tl_all, title)
        tl_columns = tl_entry.get("columns") or tl_entry.get("timeline_variables") or []

        # Description & schema
        det_all = self._detailed(project)
        desc_text = _desc_for_title(det_all, title)
        ts_all = self._trial_schema(project)
        ts_entry = _by_title(ts_all, title)
        headers = ts_entry.get("headers") or []
        columns_meta = ts_entry.get("columns") or {}

        # INIT DOCS + EXAMPLES for only used stimuli
        init_docs = _load_pkg_init_docs()
        examples = _load_pkg_examples()
        init_docs_txt = _init_docs_for_stimuli(stimuli_order, init_docs)
        examples_txt = _examples_for_stimuli(stimuli_order, examples)

        # ------------- JSON schema for LLM output -------------
        out_schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "programs": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Program"},
                }
            },
            "required": ["programs"],
            "$defs": {
                "Program": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "experiment_title": {"type": "string"},
                        "slug": {"type": "string"},
                        "file_name": {"type": "string"},
                        "python_code": {"type": "string"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["experiment_title", "slug", "file_name", "python_code", "notes"],
                }
            },
        }

        # ------------- Prompt -------------
        prompt = (
            f"EXPERIMENT TITLE: {title}\n\n"
            "DESCRIPTION (from paper):\n"
            f"{desc_text}\n\n"
            "TRIAL SCHEMA:\n"
            f"headers: {headers}\n"
            f"columns+samples: {json.dumps([{ 'name':k, **(v or {}) } for k,v in columns_meta.items()], ensure_ascii=False)[:5000]}\n\n"
            "SELECTED TIMELINE COLUMNS (use EXACTLY these when building the timeline):\n"
            f"{json.dumps(tl_columns, ensure_ascii=False)}\n\n"
            "REFINED MAPPING PLAN (units in order, with fixed_params/per_trial names):\n"
            f"{json.dumps(units, ensure_ascii=False)[:6000]}\n\n"
            "PARAM FUNCTIONS (paste EXACTLY into the file; then bind with FunctionVariable(name, fct=..., args=[...])):\n"
            f"```python\n{functions_text}\n```\n\n"
            "BINDINGS (for your reference when wiring FunctionVariables):\n"
            f"{json.dumps(fn_bindings, ensure_ascii=False)}\n\n"
            "INIT DOCS for USED STIMULI (authoritative constructor signatures + docstrings):\n"
            f"{init_docs_txt}\n\n"
            "PACKAGE EXAMPLES (only for used stimuli):\n"
            f"{examples_txt}\n\n"
            "DRAFT PROGRAM (YOU MUST FIX & COMPLETE — keep the imports, the loader call, and function names):\n"
            f"{draft_code}\n\n"
            "Task:\n"
            "  • Replace all '...' and TODOs. Paste the provided mapping functions and bind them as:\n"
            "      FunctionVariable(name='<param>', fct=<fn>, args=[...])  # use `fct`, not `func`\n"
            "    using the given BINDINGS (args must match the CSV headers in order).\n"
            "  • Define constants from fixed_params (convert types properly; preserve CSS units like '4px').\n"
            "  • Instantiate stimuli in the 'units' order and pass only parameters allowed by the INIT DOCS.\n"
            "    Match list/scalar shapes exactly as shown in the PACKAGE EXAMPLES.\n"
            "  • Use exactly the provided timeline columns. If the draft declares COLUMNS, set that; otherwise\n"
            "    create a local `columns` list and pass it to the loader exactly as in the draft.\n"
            "  • Do NOT modify the import section. Do NOT implement/rename the CSV→timeline loader; just call it.\n"
            "  • Remove any extraneous scaffolds that are unnecessary given the provided mapping functions\n"
            "    (e.g., ad-hoc mapping engines or MAPPING_* dicts present in the draft).\n"
            "  • Keep the exact signature: def get_experiment(path): ... return experiment.\n"
            "  • Do NOT include any __main__ block or prints.\n"
            "Return JSON per schema with one Program containing the full Python file."
        )

        # ------- Call LLM and normalize output -------
        out = _generate_json(prompt, out_schema) or {"programs": []}
        progs = out.get("programs") or []

        # If nothing returned, surface a minimal record
        if not progs:
            return {
                "experiment_title": title,
                "slug": slug,
                "file_name": file_name_default,
                "python_code": "",
                "notes": ["No program returned by LLM."],
                "python_file": "",
            }

        # We expect exactly one program per experiment; if multiple, take the first
        prog = dict(progs[0] or {})
        prog["experiment_title"] = title
        prog["slug"] = slug
        prog["file_name"] = prog.get("file_name") or file_name_default

        # Write the program file now (no finalize)
        code = (prog.get("python_code") or "").rstrip() + "\n"
        out_dir = project.artifacts_dir / "sweetbean" / "programs"
        out_dir.mkdir(parents=True, exist_ok=True)
        py_path = out_dir / (prog["file_name"])
        if code.strip():
            _write_text(py_path, code)
            prog["python_file"] = str(py_path.relative_to(project.artifacts_dir))
        else:
            prog["python_file"] = ""  # keep empty if no code to write

        return prog
