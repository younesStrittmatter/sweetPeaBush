# sweetExtract/steps/build_sb_program_draft.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

# Optional deps if present
try:
    from sweetExtract.steps.llm_select_timeline_variables import LLMSelectTimelineVariables  # noqa: F401
    _HAS_TLV_STEP = True
except Exception:
    _HAS_TLV_STEP = False

try:
    from sweetExtract.steps.llm_build_param_functions_positional import LLMBuildParamFunctionsPositional  # noqa: F401
    _HAS_FN_STEP = True
except Exception:
    _HAS_FN_STEP = False

# ---------- IO ----------
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

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "-", str(s or "").strip())
    s = re.sub(r"[^a-zA-Z0-9\-_.]", "", s)
    return s.lower()[:120] or "exp"

def _snake(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()

# ---------- code builders ----------
def _build_imports(stimuli: List[str]) -> str:
    uniq, seen = [], set()
    for s in stimuli:
        if s and s not in seen:
            uniq.append(s); seen.add(s)
    stim_imports = ", ".join(uniq) if uniq else ""
    header = [
        "# Imports",
        "from sweetbean import Experiment, Block",
        "from sweetbean.util import timeline_from_csv",
        f"from sweetbean.stimulus import {stim_imports}" if stim_imports else "from sweetbean.stimulus import *",
        "from sweetbean.variable import FunctionVariable, TimelineVariable, SharedVariable, DataVariable",
        "",
    ]
    return "\n".join(header)

def _columns_for_title(tlvars_idx: Dict[str, Any], title: str) -> List[str]:
    tnorm = title.strip().lower()
    for it in (tlvars_idx.get("items") or []):
        nm = (it.get("experiment_title") or it.get("title") or it.get("slug") or "").strip().lower()
        if nm == tnorm:
            return list(it.get("timeline_variables") or it.get("columns") or [])
    return []

def _functions_for_title(fn_idx: Dict[str, Any], title: str) -> List[Dict[str, Any]]:
    tnorm = title.strip().lower()
    for it in (fn_idx.get("items") or []):
        nm = (it.get("experiment_title") or it.get("title") or it.get("slug") or "").strip().lower()
        if nm == tnorm:
            return list(it.get("functions") or [])
    return []

def _units_for_title(plan_idx: Dict[str, Any], title: str) -> List[Dict[str, Any]]:
    tnorm = title.strip().lower()
    for it in (plan_idx.get("items") or []):
        nm = (it.get("experiment_title") or it.get("title") or it.get("slug") or "").strip().lower()
        if nm == tnorm:
            return list(it.get("units") or [])
    return []

def _build_functions_section(functions: List[Dict[str, Any]]) -> str:
    """
    Emits mapping function definitions (positional-arg form) and
    FunctionVariable bindings using fct=<fn>, args=[...].
    """
    lines: List[str] = []
    if not functions:
        return "\n".join([
            "    # function variables",
            "    # TODO: add generated mapping functions here (none found yet)",
            "",
        ])

    lines.append("    # function variables")
    lines.append("")  # no adapter; use fct/args directly

    for f in functions:
        code = f.get("python_code") or ""
        args = list(f.get("args") or [])
        param = f.get("param") or "param"
        fn_name = f.get("fn_name") or f"map_{_snake(param)}"
        var_name = f"var_{_snake(param)}"
        # indent the mapping function to live inside get_experiment
        indented = "\n".join(("    " + ln) for ln in code.splitlines())
        lines.append(indented)
        args_list = "[" + ", ".join(json.dumps(a) for a in args) + "]"
        lines.append(f"    {var_name} = FunctionVariable(name={json.dumps(param)}, fct={fn_name}, args={args_list})")
        lines.append("")

    return "\n".join(lines)

def _build_stimuli_placeholders(units: List[Dict[str, Any]]) -> str:
    lines, counts = [], {}
    for u in units:
        stim = u.get("stimulus") or ""
        base = _snake(stim)
        counts[base] = counts.get(base, 0) + 1
        name = f"{base}_{counts[base]}"
        lines.append(f"    {name} = {stim}(...)  # TODO: fill parameters and bind FunctionVariables where needed")
    lines.append("")
    return "\n".join(lines)

def _build_block_sequence(units: List[Dict[str, Any]]) -> str:
    names, counts = [], {}
    for u in units:
        stim = u.get("stimulus") or ""
        base = _snake(stim)
        counts[base] = counts.get(base, 0) + 1
        names.append(f"{base}_{counts[base]}")
    inside = ", ".join(names)
    return f"    block = Block([{inside}], timeline=timeline)"

# ---------- Step ----------
class BuildSBProgramDraft(BaseStep):
    """
    Deterministically materializes a draft SweetBean program (and a small JSON spec) per experiment,
    using existing artifacts only (no LLM here).

    Reads:
      - artifacts/meta/llm_map_sb_parameters_refined.json   (or llm_map_sb_parameters.json)
      - artifacts/meta/llm_timeline_variables.json          (columns for timeline_from_csv)
      - artifacts/meta/llm_param_functions.json             (positional-arg mapping functions; optional)

    Writes:
      - artifacts/sweetbean/programs/draft_<slug>.py
      - artifacts/sweetbean/programs/draft_<slug>.json
      - artifacts/meta/sb_program_drafts.json (index)
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool = False):
        super().__init__(
            name="build_sb_program_draft",
            artifact="meta/sb_program_drafts.json",
            depends_on=[FilterEmpiricalExperiments]
            + ([LLMSelectTimelineVariables] if _HAS_TLV_STEP else [])
            + ([LLMBuildParamFunctionsPositional] if _HAS_FN_STEP else []),
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # loaders
    def _plan(self, project: Project) -> Dict[str, Any]:
        meta = project.artifacts_dir / "meta"
        p = meta / "llm_map_sb_parameters_refined.json"
        if p.exists(): return _read_json(p) or {}
        q = meta / "llm_map_sb_parameters.json"
        if q.exists(): return _read_json(q) or {}
        raise FileNotFoundError("Missing refined mapping: meta/llm_map_sb_parameters_refined.json (or llm_map_sb_parameters.json)")

    def _tlvars(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_timeline_variables.json") or {"items": []}

    def _param_funcs(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_param_functions.json") or {"items": []}

    def compute_one(self, project: Project, item: Dict[str, Any], idx: int,
                    all_items: List[Dict[str, Any]], prior_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:

        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        plan      = self._plan(project)
        tlvars    = self._tlvars(project)
        fn_store  = self._param_funcs(project)

        units = _units_for_title(plan, title)
        stimuli = [u.get("stimulus") for u in units if u.get("stimulus")]

        imports = _build_imports(stimuli)

        columns = _columns_for_title(tlvars, title)
        columns_lit = ", ".join(json.dumps(c) for c in columns)
        columns_line = f"    columns = [{columns_lit}]"

        functions = _functions_for_title(fn_store, title)
        funcs_section = _build_functions_section(functions)

        stimuli_section = _build_stimuli_placeholders(units)
        block_line = _build_block_sequence(units)

        # assemble code (NO __main__)
        code_lines: List[str] = []
        code_lines.append(imports)
        code_lines.append("")
        code_lines.append("def get_experiment(path):")
        code_lines.append("    # Timeline")
        code_lines.append(columns_line)
        code_lines.append("    timeline = timeline_from_csv(path, columns)")
        code_lines.append("")
        code_lines.append(funcs_section)
        code_lines.append("    # other variables (shared + data)")
        code_lines.append("    # TODO: define any static constants or derived shared values here")
        code_lines.append("")
        code_lines.append("    # stimuli")
        code_lines.append(stimuli_section)
        code_lines.append("    # build experiment")
        code_lines.append(block_line)
        code_lines.append("    experiment = Experiment([block])")
        code_lines.append("    return experiment")
        code_lines.append("")

        code = "\n".join(code_lines)

        out_dir = project.artifacts_dir / "sweetbean" / "programs"
        out_dir.mkdir(parents=True, exist_ok=True)
        py_path = out_dir / f"draft_{slug}.py"
        json_path = out_dir / f"draft_{slug}.json"

        py_path.write_text(code, encoding="utf-8")

        spec = {
            "experiment_title": title,
            "slug": slug,
            "file_name": py_path.name,
            "columns": columns,
            "stimuli_order": stimuli,
            "has_functions": bool(functions),
            "notes": [
                "Placeholders '...' left in stimuli constructors as requested.",
                "FunctionVariable bindings use fct=<fn>, args=[headers] (no row-adapter).",
            ],
        }
        json_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "experiment_title": title,
            "slug": slug,
            "python_file": str(py_path.relative_to(project.artifacts_dir)),
            "json_file": str(json_path.relative_to(project.artifacts_dir)),
        }

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        idx_path = project.artifacts_dir / self.artifact
        _write_json(idx_path, {"programs": results})
        return {"programs": results, "index": str(idx_path.relative_to(project.artifacts_dir))}
