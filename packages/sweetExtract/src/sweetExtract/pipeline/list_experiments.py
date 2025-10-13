import json
from sweetExtract.llm_client import chat_json
from sweetExtract.config import OPENAI_MODEL_LIST, MAX_CONTEXT_CHARS
from sweetExtract.utils.json_safety import loads_loose

SYSTEM = (
  "You are an expert in experimental psychology. "
  "Identify each experiment/study. Return STRICT JSON only."
)

USER_TMPL = """Read the text and return:

{{
  "experiments": [
    {{ "id": "Experiment 1", "title": "optional or ''", "descriptor": "one-line cue" }}
  ]
}}

Rules:
- Include every experiment/study block.
- Keep descriptor short (purpose/theme).
- No markdown, no commentary.

TEXT:
{paper_text}
"""

def run(paper_text: str) -> list[dict]:
    user = USER_TMPL.format(paper_text=paper_text[:MAX_CONTEXT_CHARS])
    content = chat_json(OPENAI_MODEL_LIST, SYSTEM, user)
    data = loads_loose(content) or {"experiments": []}
    return data.get("experiments", [])
