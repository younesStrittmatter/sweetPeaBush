import json
from sweetExtract.llm_client import chat_json
from sweetExtract.config import OPENAI_MODEL_DESC, MAX_CONTEXT_CHARS
from sweetExtract.utils.json_safety import loads_loose

SYSTEM = (
    "You are a meticulous scientific editor. "
    "Produce a fully self-contained description of ONE experiment. "
    "If it references other experiments, inline those details. Use 'not reported' if missing."
)

USER_TMPL = """Given the paper text and a target experiment, return:

{{
  "id": "{id}",
  "title": "{title}",
  "description": "1–2 paragraphs, fully self-contained: participants, stimuli, tasks, timing, design, measures, trial/block counts, counterbalancing, apparatus/software, exclusions/prereg if present. Resolve cross-references by inlining details. Strictly factual; use 'not reported' when absent."
}}

Target:
- id: {id}
- title: {title}
- descriptor: {descriptor}

PAPER TEXT:
{paper_text}
"""


def run(exp_stub: dict, paper_text: str) -> dict:
    user = USER_TMPL.format(
        id=exp_stub.get("id", "Experiment"),
        title=exp_stub.get("title", ""),
        descriptor=exp_stub.get("descriptor", ""),
        paper_text=paper_text[:MAX_CONTEXT_CHARS],
    )
    content = chat_json(OPENAI_MODEL_DESC, SYSTEM, user)
    data = loads_loose(content) or {}
    return data
