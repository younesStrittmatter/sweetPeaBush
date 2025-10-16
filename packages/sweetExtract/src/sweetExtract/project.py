from __future__ import annotations
import shutil, json, re
from typing import Union, Type
from sweetExtract.steps.base import BaseStep
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterable

class Project:
    def __init__(self, root: Path, pdf_path: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.data_raw_dir = self.artifacts_dir / "data_raw"
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)

        self.pdfs_dir = self.root / "pdfs"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(pdf_path)
        self.pdf_path = self.pdfs_dir / "paper.pdf"
        shutil.copy2(pdf_path, self.pdf_path)

        self._steps: List["BaseStep"] = []
        self._by_name: Dict[str, "BaseStep"] = {}

    # --- artifact helpers ---
    def artifact_path(self, rel: str) -> Path:
        return (self.artifacts_dir / rel).resolve()

    def read_artifact(self, rel: str) -> Any:
        p = self.artifact_path(rel)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_artifact(self, rel: str, data: Any) -> None:
        p = self.artifact_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

    def has_artifact(self, rel: str) -> bool:
        return self.artifact_path(rel).exists()

    # --- helper: does artifacts/data_raw contain any real files? ---
    def has_data_raw_files(self) -> bool:
        data_raw = self.artifacts_dir / "data_raw"
        if not data_raw.exists():
            return False
        for p in data_raw.rglob("*"):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("."):
                continue
            if name in {".DS_Store"}:
                continue
            if name == ".gitkeep" and p.stat().st_size == 0:
                continue
            return True
        return False

    # --- steps registry / running ---
    def add_steps(self, steps: Iterable["BaseStep"]) -> None:
        for s in steps:
            if s.name in self._by_name:
                raise ValueError(f"Duplicate step name: {s.name}")
            self._steps.append(s)
            self._by_name[s.name] = s

    def get_step(self, name: str) -> "BaseStep":
        return self._by_name[name]

    def require(self, dep: Union[str, Type[BaseStep], BaseStep]):
        """
        - str: treat as artifact filename (e.g., 'studies.json')
        - BaseStep class/instance: ensure that step has run
        """
        if isinstance(dep, str):
            p = self.artifacts_dir / dep
            if not p.exists():
                raise FileNotFoundError(f"Required artifact missing: {p}")
            return

        # step class or instance
        self.run_step(dep if isinstance(dep, BaseStep) else dep(), force=False)

    def run_step(self, step: BaseStep, force: bool = False):
        step.run(self, force=force)

    def run_all(self, force: bool = False) -> None:
        print(f"Running all steps for project {self.root}...")
        for step in self._steps:
            print(f"--- Running step: {step.name} ---")
            step.run(self, force=force)

    def run_until(self, name: str, force: bool = False) -> None:
        for step in self._steps:
            step.run(self, force=force)
            if step.name == name:
                break

    @staticmethod
    def _norm_name(s: Optional[str]) -> str:
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    @staticmethod
    def _read_json_file(p: Path) -> Any:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _build_desc_index(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build {by_title, alias_index} over items, using normalized keys."""
        by_title: Dict[str, Dict[str, Any]] = {}
        alias_index: Dict[str, Dict[str, Any]] = {}

        for it in items or []:
            title = (it.get("title") or it.get("experiment_title") or "").strip()
            ntitle = Project._norm_name(title)
            if ntitle:
                by_title[ntitle] = it

            aliases = it.get("aliases") or []
            if isinstance(aliases, list):
                for a in aliases:
                    na = Project._norm_name(str(a))
                    if na:
                        alias_index[na] = it

            # Also index the title itself as an alias for convenience
            if ntitle and ntitle not in alias_index:
                alias_index[ntitle] = it

        return {"items": items or [], "by_title": by_title, "alias_index": alias_index}

    # ------------------------- public API -------------------------

    def load_experiment_descriptions(self) -> Dict[str, Any]:
        """
        Load detailed experiment descriptions (from DescribeExperiments).
        Returns dict with:
          - items: the raw list of experiment dicts
          - by_title: {normalized_title -> item}
          - alias_index: {normalized_alias_or_title -> item}
        """
        p = self.artifacts_dir / "meta" / "experiments_detailed.json"
        obj = self._read_json_file(p) or {}
        items = obj.get("items") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            items = []
        return self._build_desc_index(items)

    def load_empirical_descriptions(self) -> Dict[str, Any]:
        """
        Load the filtered empirical-only descriptions produced by FilterEmpiricalExperiments.
        Same shape as load_experiment_descriptions().
        """
        p = self.artifacts_dir / "meta" / "experiments_empirical_detailed.json"
        obj = self._read_json_file(p) or {}
        items = obj.get("items") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            items = []
        return self._build_desc_index(items)

    def resolve_experiment_by_name(
        self, name: str, *, empirical_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve an experiment dict by title/alias (normalized). If empirical_only=True,
        search the empirical set; otherwise search the full set.
        """
        idx = self.load_empirical_descriptions() if empirical_only else self.load_experiment_descriptions()
        key = self._norm_name(name)
        return idx["alias_index"].get(key) or idx["by_title"].get(key)
