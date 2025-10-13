# sweetExtract/steps/catalog_for_llm.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweetExtract.steps.base import BaseStep
from sweetExtract.project import Project
from sweetExtract.steps.catalog import Catalog  # ensure upstream order


class CatalogForLLM(BaseStep):
    """
    Create a slimmed LLM-facing catalog at artifacts/meta/catalog_llm.json from
    artifacts/meta/catalog.json. Keeps ONLY:
      - relpath: str
      - sheet: str | None      (Excel -> one entry per sheet; CSV/TXT -> None)
      - columns_head: list[str]  (preview only; capped)

    No file I/O besides reading meta/catalog.json. Does not re-parse data.
    """

    def __init__(self, max_columns_head: int = 32, max_items: int = 2000, force: bool = False):
        super().__init__(
            name="catalog_for_llm",
            artifact="meta/catalog_llm.json",
            depends_on=[Catalog],   # must follow Catalog step
            map_over=None,
        )
        self.max_columns_head = int(max_columns_head)
        self.max_items = int(max_items)
        self._force = bool(force)

    def should_run(self, project: Project) -> bool:
        src = project.artifacts_dir / "meta" / "catalog.json"
        dst = project.artifacts_dir / "meta" / "catalog_llm.json"
        # Run only if source exists AND (forced OR output missing)
        return src.exists() and (self._force or not dst.exists())

    def _from_text_table(self, f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        relpath = f.get("relpath")
        if not relpath:
            return None
        cols = list(f.get("columns_head") or [])
        return {
            "relpath": relpath,
            "sheet": None,
            "columns_head": cols[: self.max_columns_head],
        }

    def _from_excel(self, f: Dict[str, Any]) -> List[Dict[str, Any]]:
        relpath = f.get("relpath")
        out: List[Dict[str, Any]] = []
        if not relpath:
            return out
        sheet_headers = f.get("sheet_headers") or {}
        if isinstance(sheet_headers, dict) and sheet_headers:
            for sheet_name, header_cols in sheet_headers.items():
                out.append({
                    "relpath": relpath,
                    "sheet": str(sheet_name),
                    "columns_head": list(header_cols or [])[: self.max_columns_head],
                })
        else:
            # Keep stub if we couldn't peek headers/sheets
            out.append({
                "relpath": relpath,
                "sheet": None,
                "columns_head": [],
            })
        return out

    def compute(self, project: Project) -> Dict[str, Any]:
        src_path = project.artifacts_dir / "meta" / "catalog.json"
        obj = json.loads(src_path.read_text(encoding="utf-8"))

        files = obj.get("files") or []
        out_items: List[Dict[str, Any]] = []

        for f in files:
            kind = (f.get("kind") or "").lower()
            if kind not in {"table/text", "table/excel"}:
                continue

            if kind == "table/text":
                rec = self._from_text_table(f)
                if rec:
                    out_items.append(rec)

            elif kind == "table/excel":
                out_items.extend(self._from_excel(f))

            if len(out_items) >= self.max_items:
                break

        return {
            "files": out_items,
            "caps": {
                "max_items": self.max_items,
                "max_columns_head": self.max_columns_head,
            },
            "notes": [
                "Slimmed catalog for LLM only; derived from meta/catalog.json without re-reading data.",
                "Excel files are expanded per sheet using pre-parsed sheet_headers.",
            ],
        }
