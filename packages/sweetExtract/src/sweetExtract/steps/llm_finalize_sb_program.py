# sweetExtract/steps/llm_finalize_sb_program.py
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments
from sweetExtract.steps.build_sb_program_draft import BuildSBProgramDraft

def _read_json(p: Path)->Any:
    try:
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    except Exception: pass
    return None

def _write_json(p: Path, obj: Any)->None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _slug(s:str)->str:
    s=re.sub(r"\s+","-",str(s or "").strip()); s=re.sub(r"[^a-zA-Z0-9\-_.]","",s); return s.lower()[:120] or "exp"

def _gen(prompt:str, schema:Dict[str,Any])->Dict[str,Any]:
    try:
        from sweetExtract.utils.llm_client import generate_response
    except Exception:
        return {"programs":[]}
    return generate_response(
        model=os.getenv("SWEETEXTRACT_LLM_MODEL","gpt-5"),
        system_prompt=(
            "You are given a DRAFT SweetBean program that MUST keep this scaffold:\n"
            "  • COLUMNS = [...]\n"
            "  • plain constants (e.g., fixation_duration = 500)\n"
            "  • mapping functions def map_<param>(row): return _eval(MAPPING_<PARAM>, row)\n"
            "  • FunctionVariable(name='<param>', func=map_<param>)\n"
            "  • Stimuli wiring using constants + FunctionVariables\n"
            "  • def get_experiment(path): timeline = load_timeline_from_csv(path, COLUMNS)\n"
            "Your job: ONLY fix small issues and polish; DO NOT change the scaffold or introduce new dependencies. "
            "Do not invent column/parameter names. Return JSON exactly per schema; put the final Python in 'python_code'."
        ),
        prompt=prompt,
        json_schema=schema,
        schema_name="LLM_FinalizeSBProgram_KeepScaffold_V2",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    ) or {"programs":[]}

class LLMFinalizeSBProgram(BaseStep):
    artifact_is_list=False
    default_array_key=None
    def __init__(self, force: bool=False):
        super().__init__(
            name="llm_finalize_sb_program",
            artifact="meta/llm_sb_program_spec.json",
            depends_on=[BuildSBProgramDraft, FilterEmpiricalExperiments],
            map_over=FilterEmpiricalExperiments,
        )
        self._force=bool(force)

    def should_run(self, project: Project)->bool:
        out=project.artifacts_dir / self.artifact
        return True if self._force else not out.exists()

    # loaders
    def _draft(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "llm_sb_program_draft.json") or {}
    def _schema(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "sb_trial_schema_for_llm.json") or {}
    def _detailed(self, project: Project)->Dict[str,Any]:
        return _read_json(project.artifacts_dir / "meta" / "experiments_empirical_detailed.json") or {}

    def compute_one(self, project: Project, item: Dict[str,Any], idx:int,
                    all_items: List[Dict[str,Any]], prior: List[Dict[str,Any]])->Dict[str, Any]:
        title=(item or {}).get("title") or f"Experiment {idx+1}"
        slug=_slug(title)

        draft_all = self._draft(project)
        prog = next((p for p in (draft_all.get("programs") or []) if (p.get("experiment_title")==title or p.get("slug")==slug)), None) or {}
        draft_code = prog.get("python_code_draft","")
        default_file = f"{slug}.py"

        if not draft_code.strip():
            # nothing to finalize; emit empty errata so the materializer can skip/write draft later
            return {"programs":[{
                "experiment_title": title,
                "slug": slug,
                "file_name": default_file,
                "python_code": "",
                "notes": ["No draft code available; finalize step left empty."]
            }]}

        detailed = self._detailed(project)
        desc = ""
        for e in (detailed.get("items") or []):
            t = (e.get("title") or "").strip().lower()
            aliases = [a.strip().lower() for a in (e.get("aliases") or [])]
            if t == title.strip().lower() or title.strip().lower() in aliases:
                desc = (e.get("standalone_summary") or e.get("description") or e.get("methods_text") or "").strip()
                break

        schema = {
            "type":"object","additionalProperties":False,
            "properties":{"programs":{"type":"array","items":{"$ref":"#/$defs/P"}}},
            "required":["programs"],
            "$defs":{"P":{"type":"object","additionalProperties":False,
                          "properties":{"experiment_title":{"type":"string"},
                                        "slug":{"type":"string"},
                                        "file_name":{"type":"string"},
                                        "python_code":{"type":"string"},
                                        "notes":{"type":"array","items":{"type":"string"}}},
                          "required":["experiment_title","slug","file_name","python_code","notes"]}}
        }

        prompt = (
            f"EXPERIMENT: {title}\n\n"
            "DESCRIPTION:\n" + desc + "\n\n"
            "DRAFT CODE (KEEP THE SCAFFOLD; fix only):\n" + draft_code + "\n"
        )
        out = _gen(prompt, schema) or {"programs":[]}

        progs = out.get("programs") or []
        if not progs or not (progs[0].get("python_code") or "").strip():
            # robust fallback to draft
            return {"programs":[{
                "experiment_title": title,
                "slug": slug,
                "file_name": default_file,
                "python_code": draft_code,
                "notes": ["LLM returned empty; using draft verbatim."]
            }]}

        fixed=[]
        for p in progs:
            code = (p.get("python_code") or "").strip() or draft_code
            fixed.append({
                "experiment_title": title,
                "slug": slug,
                "file_name": p.get("file_name") or default_file,
                "python_code": code,
                "notes": p.get("notes") or []
            })
        return {"programs": fixed}

    def finalize(self, project: Project, results: List[Dict[str,Any]])->Dict[str,Any]:
        flat=[]
        for r in results: flat.extend(r.get("programs") or [])
        path = project.artifacts_dir / self.artifact
        _write_json(path, {"programs": flat})
        return {"programs": flat, "index": str(path.relative_to(project.artifacts_dir))}
