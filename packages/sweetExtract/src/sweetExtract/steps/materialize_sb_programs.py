from __future__ import annotations
import json, os, re, stat
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.llm_finalize_sb_program import LLMFinalizeSBProgram  # depends on the finalized spec

# ---------- IO ----------
def _read_json(p: Path) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _write_text(p: Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "-", str(s or "").strip())
    s = re.sub(r"[^a-zA-Z0-9\-_.]", "", s)
    return s.lower()[:120] or "exp"

def _programs_from_spec(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    # support both flattened {"programs":[...]} and multi-item {"items":[{"programs":[...]}...]}
    if isinstance(obj.get("programs"), list):
        return [p for p in obj["programs"] if isinstance(p, dict)]
    progs: List[Dict[str, Any]] = []
    for it in (obj.get("items") or []):
        for p in (it or {}).get("programs") or []:
            if isinstance(p, dict):
                progs.append(p)
    return progs

def _find_trial_schema_entry(schema: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
    tnorm = (title or "").strip().lower()
    for it in (schema.get("items") or []):
        nm = (it.get("experiment_title") or it.get("title") or "").strip().lower()
        if nm == tnorm:
            return it
    return None

def _patch_data_path_in_code(code: str, data_path: str) -> str:
    """
    Best-effort: if the generated program contains a SPEC/BUILD_SPEC dict with "data_path": "",
    replace it with the provided path. This keeps the script runnable without CLI args.
    """
    if not data_path:
        return code
    # very conservative replace: match "data_path": "" with optional whitespace
    return re.sub(r'("data_path"\s*:\s*)"[^"]*"', r'\1' + json.dumps(data_path), code, count=2)

class MaterializeSBPrograms(BaseStep):
    """
    Reads meta/llm_sb_program_spec.json and writes runnable Python files.

    Inputs:
      - meta/llm_sb_program_spec.json   (from LLMFinalizeSBProgram)
      - meta/sb_trial_schema_for_llm.json (optional; if present and entry has data_path we patch it into code)

    Writes:
      - sweetbean/programs/<file_name>     (Python scripts, executable if shebang present)
      - meta/sb_programs_written.json      (index of written files)
    """
    artifact_is_list = False
    default_array_key = None

    def __init__(self, force: bool=False):
        super().__init__(
            name="materialize_sb_programs",
            artifact="meta/sb_programs_written.json",
            depends_on=[LLMFinalizeSBProgram, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out = project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # ---------- loaders ----------
    def _spec_path_candidates(self, project: Project) -> List[Path]:
        meta = project.artifacts_dir / "meta"
        # primary file name used by the pipeline
        cands = [meta / "llm_sb_program_spec.json"]
        # tolerate a common typo without blocking
        cands.append(meta / "lm_sb_program_spec.json")
        return cands

    def _load_program_spec(self, project: Project) -> Dict[str, Any]:
        for p in self._spec_path_candidates(project):
            obj = _read_json(p)
            if obj:
                return obj
        return {}

    def _load_trial_schema(self, project: Project) -> Dict[str, Any]:
        return _read_json(project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json") or {}

    # ---------- core ----------
    def compute_one(self, project: Project, item: Dict, idx: int,
                    all_items: List[Dict], prior_outputs: List[Dict]) -> Dict[str, Any]:
        title = (item or {}).get("title") or f"Experiment {idx+1}"
        slug  = _slug(title)

        prog_spec = self._load_program_spec(project)
        programs  = _programs_from_spec(prog_spec)
        if not programs:
            # Nothing to write for this project run
            return {"experiment_title": title, "slug": slug, "written": []}

        trial_schema = self._load_trial_schema(project)

        out_dir = project.artifacts_dir / "sweetbean" / "programs"
        written: List[Dict[str, str]] = []

        for prog in programs:
            pt = (prog.get("experiment_title") or "").strip()
            if pt.lower() != title.strip().lower() and (prog.get("slug") or "") != slug:
                # This program belongs to a different experiment in the set
                continue

            file_name = prog.get("file_name") or f"{slug}.py"
            code = prog.get("python_code") or ""
            if not code.strip():
                # Skip empty code blobs
                continue

            # If we have a data_path in the trial schema, patch it into code when a SPEC/BUILD_SPEC block exists
            ts = _find_trial_schema_entry(trial_schema, title) or {}
            data_path = ts.get("data_path") or ts.get("file") or ""
            patched = _patch_data_path_in_code(code, data_path)

            out_path = out_dir / file_name
            _write_text(out_path, patched)

            # Make executable if it starts with a shebang
            if patched.lstrip().startswith("#!"):
                mode = os.stat(out_path).st_mode
                os.chmod(out_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            written.append({
                "experiment_title": title,
                "slug": slug,
                "file": str(out_path.relative_to(project.artifacts_dir)),
                "data_path_patched": data_path or ""
            })

        return {"experiment_title": title, "slug": slug, "written": written}

    def finalize(self, project: Project, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        flat: List[Dict[str, str]] = []
        for r in results:
            flat.extend(r.get("written") or [])
        idx_path = project.artifacts_dir / self.artifact
        _write_text(idx_path, json.dumps({"programs_written": flat}, ensure_ascii=False, indent=2))
        return {"programs_written": flat, "index": str(idx_path.relative_to(project.artifacts_dir))}
