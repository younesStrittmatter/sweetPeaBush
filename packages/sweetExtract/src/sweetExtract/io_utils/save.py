from pathlib import Path
import json


def save_json(data, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown(experiments: list[dict], path: str | Path):
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Extracted Experiments", ""]
    for e in experiments:
        title = f": {e['title']}" if e.get("title") else ""
        lines += [f"## {e.get('id', 'Experiment')}{title}", "", (e.get("description") or "").strip(), ""]
    p.write_text("\n".join(lines), encoding="utf-8")
