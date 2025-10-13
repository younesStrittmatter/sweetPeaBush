#!/usr/bin/env python3
"""
Builds a compact prompt bank for SweetBean stimuli in three slices, source-first, without cloning.

Ground truth: scan `src/sweetbean/stimulus/*.py` for stimulus classes (and aliases like
`ROK = RandomObjectKinematogram`). For each class, find its docs page under
`docs/Stimuli/<dir>/index.md`, extract the **first non-empty line after the H1** as the
short description, and collect example `*.py` files in that docs directory that actually
reference the class or its alias(es).

Outputs to ./prompt_bank/ :
  1) step1_stimuli_index.json  – list of {key, title, description, docs_index_relpath}
  2) step2_examples.json       – {key -> [ {filename, relpath, code} ]}
  3) step3_init_docs.json      – {key -> {class_name, signature, init_docstring, source_relpath}}
  4) manifest.json             – build metadata (GitHub repo/ref)

Repo/ref are hardcoded to AutoResearch/sweetbean @ main.
"""
from __future__ import annotations

import ast
import base64
import datetime as dt
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm
import io
import tarfile

# ----------------------------- Config & helpers ------------------------------

OWNER_REPO = "AutoResearch/sweetbean"
GIT_REF = "main"

STIM_DOCS_ROOT = Path("docs") / "Stimuli"
STIM_SRC_ROOT = Path("src") / "sweetbean" / "stimulus"

# Optional hints for ambiguous mapping (extend as needed)
DOCS_TO_CLASSES_HINTS: Dict[str, List[str]] = {
    "rok": ["RandomObjectKinematogram", "ROK"],
    "rdp": ["RandomDotPatterns", "RDP", "ROK"],
}

MD_H1 = re.compile(r"^\s*#\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
MD_FIRST_H2 = re.compile(r"^##\s+", flags=re.MULTILINE)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# ----------------------------- Markdown parsing -----------------------------

def parse_index_md(md_path: Path) -> Tuple[str, str]:
    text = md_path.read_text(encoding="utf-8")
    # Normalize newlines and strip BOM if present
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    # Extract H1 title
    m = MD_H1.search(text)
    title = m.group("title").strip() if m else md_path.parent.name

    # Slice after H1 line (or whole file if no H1)
    after_h1 = text[m.end():] if m else text

    # Keep only content before the first '##' header (top-level description block)
    h2 = MD_FIRST_H2.search(after_h1)
    top_block = after_h1[:h2.start()] if h2 else after_h1

    # Remove any leading blank or heading lines (defensive against stray '# ...')
    lines = top_block.split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    cleaned_block = "\n".join(lines[i:])

    description = cleaned_block.strip()
    return title, description


# ----------------------------- AST parsing ----------------------------------

@dataclass
class InitInfo:
    class_name: Optional[str]
    signature: Optional[str]
    init_docstring: Optional[str]
    source_relpath: Optional[str]


@dataclass
class ClassInfo:
    class_name: str
    module_relpath: str
    signature: Optional[str]
    init_docstring: Optional[str]
    aliases: List[str]


def _ast_param_to_str(arg: ast.arg, default: Optional[ast.AST]) -> str:
    name = arg.arg
    if default is None: return name
    try:
        val = ast.unparse(default)
    except Exception:
        val = "..."
    return f"{name}={val}"


def _build_signature_from_args(args: ast.arguments) -> str:
    parts: List[str] = []
    reg_args = list(args.args)
    if reg_args and reg_args[0].arg == "self":
        reg_args = reg_args[1:]
    n_defaults = len(args.defaults)
    default_start = len(args.args) - n_defaults if n_defaults else None
    for i, a in enumerate(reg_args):
        default = None
        if n_defaults and default_start is not None:
            global_index = i + 1
            if global_index >= default_start:
                di = global_index - default_start
                if 0 <= di < n_defaults:
                    default = args.defaults[di]
        parts.append(_ast_param_to_str(a, default))
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    if args.kwonlyargs:
        if not args.vararg:
            parts.append("*")
        for i, a in enumerate(args.kwonlyargs):
            default = args.kw_defaults[i] if args.kw_defaults else None
            parts.append(_ast_param_to_str(a, default))
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return f"__init__(self, {', '.join(parts)})" if parts else "__init__(self)"


def _is_stimulus_class(node: ast.ClassDef) -> bool:
    """Capitalized, and (has __init__ OR 'type' attr OR inherits *Stimulus)."""
    if not node.name or not node.name[0].isupper():
        return False
    has_init = any(isinstance(b, ast.FunctionDef) and b.name == "__init__" for b in node.body)
    has_type_attr = any(
        isinstance(b, ast.Assign)
        and any(getattr(t, "id", "") == "type" for t in getattr(b, "targets", []))
        for b in node.body
    )
    inherits_stimulus = False
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
        if name and name.lower().endswith("stimulus"):
            inherits_stimulus = True
            break
    return has_init or has_type_attr or inherits_stimulus


def extract_classes_from_module(module_path: Path) -> Tuple[List[ClassInfo], Dict[str, str]]:
    """Return (classes, aliases) where aliases maps ALIAS -> ClassName (e.g., ROK -> RandomObjectKinematogram)."""
    classes: List[ClassInfo] = []
    aliases: Dict[str, str] = {}
    if not module_path.exists():
        return classes, aliases
    src = module_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # alias assignments like ROK = RandomObjectKinematogram
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tgt = node.targets[0].id
            if isinstance(node.value, ast.Name) and tgt.isupper():
                aliases[tgt] = node.value.id
    # classes
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_stimulus_class(node):
            init_func = next((b for b in node.body if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
            signature = _build_signature_from_args(init_func.args) if init_func else None
            init_doc = ast.get_docstring(init_func) if init_func else None
            classes.append(
                ClassInfo(
                    class_name=node.name,
                    module_relpath=str(module_path),
                    signature=signature,
                    init_docstring=init_doc,
                    aliases=[a for a, c in aliases.items() if c == node.name],
                )
            )
    return classes, aliases


# ----------------------------- Docs + examples linking ----------------------

@dataclass
class StimulusEntry:
    key: str
    title: str
    description: str
    docs_index_relpath: Optional[str]


@dataclass
class ExampleEntry:
    filename: str
    relpath: str
    code: str


@dataclass
class StimulusRecord:
    index: StimulusEntry
    examples: List[ExampleEntry]
    init_info: InitInfo


def _build_docs_catalog(repo_root: Path) -> Dict[str, Dict[str, str]]:
    """Map multiple keys to the same docs record:
       - 'dir' name
       - slug(dir)
       - slug(H1 title)
       Value: {"title","desc","index_rel","dir"}.
    """
    out: Dict[str, Dict[str, str]] = {}
    base = repo_root / STIM_DOCS_ROOT
    if not base.exists():
        return out
    for d in sorted([p for p in base.iterdir() if p.is_dir()]):
        idx = d / "index.md"
        if not idx.exists():
            continue
        title, desc = parse_index_md(idx)
        rec = {"title": title, "desc": desc, "index_rel": str(idx.relative_to(repo_root)), "dir": d.name}
        out[d.name] = rec
        out[_slug(d.name)] = rec
        out[_slug(title)] = rec
    return out


def _best_docs_match(class_name: str, aliases: List[str], module_basename: str, catalog: Dict[str, Dict[str, str]]) -> \
Optional[Dict[str, str]]:
    cand_keys = [
        class_name, _slug(class_name),
        module_basename, _slug(module_basename),
        *aliases, *[_slug(a) for a in aliases]
    ]
    for k in cand_keys:
        if k in catalog:
            return catalog[k]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    kebab = snake.replace("_", "-")
    for k in [snake, kebab, _slug(kebab)]:
        if k in catalog:
            return catalog[k]
    for dkey, clist in DOCS_TO_CLASSES_HINTS.items():
        if class_name in clist or any(a in clist for a in aliases):
            if dkey in catalog:
                return catalog[dkey]
    return None


def _collect_examples_for_docs_dir(repo_root: Path, docs_dir_name: str, class_name: str, aliases: List[str]) -> List[
    ExampleEntry]:
    out: List[ExampleEntry] = []
    d = repo_root / STIM_DOCS_ROOT / docs_dir_name
    if not d.exists():
        return out
    toks = [class_name] + aliases
    pat = re.compile(r"\b(" + "|".join(map(re.escape, toks)) + r")\b") if toks else None
    for p in sorted(d.glob("*.py")):
        if p.name == "__init__.py":
            continue
        code = p.read_text(encoding="utf-8")
        if pat is None or pat.search(code):
            out.append(ExampleEntry(filename=p.name, relpath=str(p.relative_to(repo_root)), code=code))
    return out


# ----------------------------- Build (source-first) -------------------------

def build_prompt_bank(repo_root: Path) -> Dict[str, StimulusRecord]:
    records: Dict[str, StimulusRecord] = {}

    src_dir = repo_root / STIM_SRC_ROOT
    if not src_dir.exists():
        return records

    docs_catalog = _build_docs_catalog(repo_root)

    for module_path in sorted(src_dir.glob("*.py")):
        classes, _aliases_map = extract_classes_from_module(module_path)
        module_basename = module_path.stem
        for ci in classes:
            key = _slug(ci.class_name)
            match = _best_docs_match(ci.class_name, ci.aliases, module_basename, docs_catalog)
            if match:
                title, desc, idx_rel, docs_dir_name = match["title"], match["desc"], match["index_rel"], match["dir"]
            else:
                title, desc, idx_rel, docs_dir_name = ci.class_name, "", None, None

            index_entry = StimulusEntry(key=key, title=title, description=desc, docs_index_relpath=idx_rel)

            examples: List[ExampleEntry] = []
            if docs_dir_name:
                examples = _collect_examples_for_docs_dir(repo_root, docs_dir_name, ci.class_name, ci.aliases)

            init_info = InitInfo(
                class_name=ci.class_name,
                signature=ci.signature,
                init_docstring=ci.init_docstring,
                source_relpath=ci.module_relpath,
            )

            records[key] = StimulusRecord(index=index_entry, examples=examples, init_info=init_info)

    return records


# ----------------------------- Output writers --------------------------------

def write_step_files(records: Dict[str, StimulusRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    step1 = [asdict(rec.index) for rec in records.values()]
    (out_dir / "stimuli_index.json").write_text(
        json.dumps(step1, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    step2: Dict[str, List[Dict[str, Any]]] = {}
    for key, rec in records.items():
        step2[key] = [asdict(ex) for ex in rec.examples]
    (out_dir / "examples.json").write_text(
        json.dumps(step2, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    step3: Dict[str, Dict[str, Any]] = {}
    for key, rec in records.items():
        step3[key] = asdict(rec.init_info)
    (out_dir / "init_docs.json").write_text(
        json.dumps(step3, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_manifest_remote(out_dir: Path) -> None:
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "github": OWNER_REPO,
        "ref": GIT_REF,
        "notes": "Built from GitHub raw files without cloning (minimal mirror).",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ----------------------------- Remote fetch helpers --------------------------


def _mirror_minimal_repo(tmp: Path) -> None:
    """
    Download a single tarball of the repo at GIT_REF and extract only:
      - src/sweetbean/stimulus/**
      - docs/Stimuli/**
    into `tmp`, preserving relative paths.
    This avoids Contents API rate limits entirely.
    """
    tar_url = f"https://codeload.github.com/{OWNER_REPO}/tar.gz/refs/heads/{GIT_REF}"
    r = requests.get(tar_url, timeout=60)
    r.raise_for_status()

    # Load tarball in-memory
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        # The tarball has a top-level folder like "<repo>-<something>/"
        # Detect that prefix from the first member.
        members = tar.getmembers()
        if not members:
            raise SystemExit("Downloaded tarball is empty.")
        top_prefix = members[0].name.split("/")[0].rstrip("/")  # "<repo>-<sha>" or similar

        # Paths we care about inside the tarball
        keep_prefixes = [
            f"{top_prefix}/src/sweetbean/stimulus/",
            f"{top_prefix}/docs/Stimuli/",
        ]

        for m in members:
            name = m.name
            # Skip directories
            if not m.isfile():
                continue
            # Keep only files under our prefixes
            if not any(name.startswith(pref) for pref in keep_prefixes):
                continue

            # Compute the relative path (strip the top folder)
            rel = name.split("/", 1)[1] if "/" in name else name
            # Write to tmp
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src_f, open(dest, "wb") as out_f:
                out_f.write(src_f.read())


# ----------------------------- Main (no args) -------------------------------

def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="sweetbean_minimal_"))
    try:
        _mirror_minimal_repo(tmpdir)
        out_dir = Path("../packages/sweetExtract/src/sweetExtract/info/sb").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        records = build_prompt_bank(tmpdir)
        if not records:
            raise SystemExit("No stimuli/classes found in remote repo — check repo/ref.")

        write_step_files(records, out_dir)
        write_manifest_remote(out_dir)

        print(f"Wrote prompt bank to: {out_dir}")
        print(f"  - step1_stimuli_index.json: {out_dir / 'stimuli_index.json'}")
        print(f"  - step2_examples.json:      {out_dir / 'examples.json'}")
        print(f"  - step3_init_docs.json:     {out_dir / 'init_docs.json'}")
        print(f"  - manifest.json:            {out_dir / 'manifest.json'}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
