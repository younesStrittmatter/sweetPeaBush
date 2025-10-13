import json, re


def loads_loose(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    cleaned = s.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned).rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
    m = re.search(r"\[[\s\S]*\]", cleaned) or re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None
