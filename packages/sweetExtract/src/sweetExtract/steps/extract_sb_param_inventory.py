from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.filter_empirical_experiments import FilterEmpiricalExperiments

# ---------- tiny io ----------
def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

@dataclass
class Param:
    name: str
    has_default: bool
    default: Optional[str]

def _parse_signature(sig: str) -> List[Param]:
    """Parse a Python-style __init__ signature string (no heuristics)."""
    if not isinstance(sig, str) or "(" not in sig or ")" not in sig:
        return []
    inside = sig[sig.find("(")+1:sig.rfind(")")]
    parts = [p.strip() for p in inside.split(",")]
    out: List[Param] = []
    for p in parts:
        if not p or p == "self":
            continue
        if "=" in p:
            name, default = p.split("=", 1)
            out.append(Param(name=name.strip(), has_default=True, default=default.strip()))
        else:
            out.append(Param(name=p.strip(), has_default=False, default=None))
    return out

class ExtractSBParamInventory(BaseStep):
    """
    Build source-of-truth inventory of every SweetBean stimulus class & its init params.

    Inputs (priority):
      1) packaged: sweetExtract.info.sb.init_docs.json (if present)
      2) local override: artifacts/meta/sb_init_docs.json

    Output:
      artifacts/meta/sb_param_inventory.json
      {
        "items": [
          {"class_name": "RSVP", "signature": "...", "doc":"...",
           "params": [{"name":"stimulus_duration","has_default":true,"default":"50"}, ...]
          }, ...
        ]
      }
    """
    artifact_is_list = False
    default_array_key = "items"

    def __init__(self, force: bool=False):
        super().__init__(
            name="extract_sb_param_inventory",
            artifact="meta/sb_param_inventory.json",
            depends_on=[FilterEmpiricalExperiments],
            map_over=None,
        )
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        out_path = project.artifacts_dir / self.artifact
        return True if self._force else not out_path.exists()

    def compute(self, project: Project) -> Dict[str, Any]:
        # 1) Try packaged resource (if your repo includes it)
        packaged = None
        try:
            from importlib.resources import files
            packaged = json.loads(files("sweetExtract.info.sb").joinpath("init_docs.json").read_text("utf-8"))
        except Exception:
            packaged = None

        # 2) Project-local override
        local = _read_json(project.artifacts_dir / "meta" / "sb_init_docs.json")

        raw = local or packaged or {}
        items: List[Dict[str, Any]] = []
        for key, rec in (raw or {}).items():
            cls = (rec or {}).get("class_name") or key
            sig = (rec or {}).get("signature") or ""
            doc = (rec or {}).get("init_docstring") or (rec or {}).get("doc") or ""
            params = [p.__dict__ for p in _parse_signature(sig)]
            items.append({"class_name": cls, "signature": sig, "doc": doc, "params": params})

        out = {"items": items}
        out_path = project.artifacts_dir / self.artifact
        _write_json(out_path, out)
        return out
