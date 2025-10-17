from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict, Union, Type
import json, shutil

from tqdm import tqdm

from sweetExtract.utils.json_helpers import json_default, to_jsonable

MapOverT = Union[
    Tuple[str, str],           # ("artifact_filename.json", "array_key")
    "BaseStep",                # another step instance
    Type["BaseStep"],          # another step class (we'll instantiate it)
]

class BaseStep:
    artifact: str
    artifact_is_list: bool = False
    default_array_key: Optional[str] = None
    _force: bool = False

    def __init__(
        self,
        name: str,
        artifact: str,
        depends_on: list[Union[str, Type["BaseStep"], "BaseStep"]],
        map_over: Optional[MapOverT] = None,
    ):
        self.name = name
        self.artifact = artifact
        self.depends_on = depends_on
        self.map_over = map_over

    # ---------- hooks for subclasses ----------
    def should_run(self, project) -> bool:
        return True

    def should_run_one(self, project, item: Dict, idx: int) -> bool:
        return True

    def compute(self, project) -> Any:
        raise NotImplementedError()

    def compute_one(self, project, item: Dict, idx: int, all_items: List[Dict], prior_outputs: List[Dict]) -> Any:
        raise NotImplementedError()

    # ---------- artifact I/O ----------
    def _artifact_path(self, project) -> Path:
        return project.artifacts_dir / self.artifact

    def _item_dir(self, project) -> Path:
        return project.artifacts_dir / self.name

    def _item_path(self, project, idx: int) -> Path:
        return self._item_dir(project) / f"{idx}.json"

    def exists(self, project) -> bool:
        return self._artifact_path(project).exists()

    def read(self, project) -> Any:
        with self._artifact_path(project).open("r", encoding="utf-8") as f:
            return json.load(f)

    def write(self, project, data: Any) -> None:
        p = self._artifact_path(project)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False, default=json_default)

    # ---------- cleanup helpers ----------
    def _purge_outputs(self, project) -> None:
        """
        Remove this step's combined artifact and per-item directory.
        Called when running with force=True to avoid stale per-item files.
        """
        # Combined artifact
        ap = self._artifact_path(project)
        if ap.exists():
            try:
                ap.unlink()
            except Exception:
                try:
                    ap.write_text("", encoding="utf-8")
                    ap.unlink(missing_ok=True)
                except Exception:
                    pass

        # Per-item directory
        idir = self._item_dir(project)
        if idir.exists():
            shutil.rmtree(idir, ignore_errors=True)

    # ---------- map source resolver ----------
    def _resolve_map_source(self, project) -> tuple[list[Dict], Optional["BaseStep"]]:
        if isinstance(self.map_over, tuple):
            src_filename, array_key = self.map_over
            src_obj = json.loads((project.artifacts_dir / src_filename).read_text(encoding="utf-8"))
            if not isinstance(src_obj, dict) or array_key not in src_obj or not isinstance(src_obj[array_key], list):
                raise ValueError(f"Expected list at key '{array_key}' in {src_filename}")
            return src_obj[array_key], None

        if isinstance(self.map_over, type) and issubclass(self.map_over, BaseStep):
            src_step: BaseStep = self.map_over()
        elif isinstance(self.map_over, BaseStep):
            src_step = self.map_over
        else:
            raise ValueError("Invalid 'map_over' type.")

        # Ensure the source step has run
        project.run_step(src_step, force=False)
        src_obj = src_step.read(project)

        if src_step.artifact_is_list:
            if not isinstance(src_obj, list):
                raise ValueError(f"{src_step.name} declared artifact_is_list=True but artifact is not a list.")
            return src_obj, src_step

        key = src_step.default_array_key
        if key is None:
            if isinstance(src_obj, dict):
                list_keys = [k for k, v in src_obj.items() if isinstance(v, list)]
                if len(list_keys) == 1:
                    key = list_keys[0]
                else:
                    raise ValueError(
                        f"Cannot determine array to map over for '{src_step.name}'. "
                        f"Set default_array_key on the source step."
                    )
            else:
                raise ValueError(f"Artifact of '{src_step.name}' must be dict when no default_array_key is set.")
        items = src_obj.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f"Expected list at key '{key}' in artifact of '{src_step.name}'.")
        return items, src_step

    # ---------- main runner ----------
    def run(self, project, force: bool = False) -> None:
        effective_force = bool(force or self._force)

        # A) Fast skip unless forcing
        if self.exists(project) and not effective_force:
            return

        # B) Ensure dependencies
        for dep in self.depends_on:
            project.require(dep)

        # C) Pre-conditions
        if not self.should_run(project):
            return

        # D) If forcing, purge outputs first
        if effective_force:
            self._purge_outputs(project)

        # E) Non-mapped steps
        if self.map_over is None:
            result = self.compute(project)
            self.write(project, result)
            return

        # F) Mapped steps
        items, _ = self._resolve_map_source(project)
        out_dir = self._item_dir(project)
        out_dir.mkdir(parents=True, exist_ok=True)

        combined: List[Dict] = []
        prior_outputs: List[Dict] = []
        processed_any = False

        for idx, item in tqdm(enumerate(items)):
            p = self._item_path(project, idx)

            # Reuse only when NOT forcing
            if p.exists() and not effective_force:
                out = json.loads(p.read_text(encoding="utf-8"))
                combined.append(out)
                prior_outputs.append(out)
                processed_any = True  # counts as produced earlier
                continue

            if not self.should_run_one(project, item, idx):
                # skip this item; do not write its file
                continue

            out = self.compute_one(project, item, idx, items, prior_outputs)
            p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            combined.append(out)
            prior_outputs.append(out)
            processed_any = True

        # Only write the combined artifact if we processed anything (now or reused).
        if processed_any:
            payload: Any = combined if self.artifact_is_list else {"items": combined}
            self.write(project, payload)
        # else: leave no artifact so the step can run later when prerequisites become true.
