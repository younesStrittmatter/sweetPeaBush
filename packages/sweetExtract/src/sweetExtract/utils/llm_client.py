from __future__ import annotations
from typing import Iterable, Union, Optional, Dict, Any, Tuple
from pathlib import Path
from copy import deepcopy
from openai import OpenAI
import json
from dotenv import load_dotenv


def generate_response(
        *,
        model: str = "gpt-5",
        system_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        file_paths: Optional[Iterable[Union[str, Path]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "schema",
        strict_schema: bool = True,
        reasoning_effort: Optional[str] = "low",
        text_verbosity: str = "low",
        include_raw: bool = False,
) -> Any | Tuple[Any, Any]:
    """
    - If `json_schema` is provided, we enforce structured JSON and return parsed data.
    - If `json_schema` is an ARRAY schema, we auto-wrap it into an OBJECT under key 'items'
      (required by the API), then unwrap before returning so the caller still gets a list.
    - If `json_schema` is None, we return plain text.
    """
    load_dotenv()
    client = OpenAI()

    # ----- Build input items -----
    input_items = []

    if system_prompt:
        input_items.append({
            "role": "developer",
            "content": [{"type": "input_text", "text": system_prompt}]
        })

    user_content = []
    if prompt:
        user_content.append({"type": "input_text", "text": prompt})

    if file_paths:
        for p in file_paths:
            p = Path(p)
            with p.open("rb") as fh:
                uploaded = client.files.create(file=fh, purpose="user_data")
            user_content.append({"type": "input_file", "file_id": uploaded.id})

    if not user_content:
        user_content = [{"type": "input_text", "text": ""}]

    input_items.append({"role": "user", "content": user_content})

    # ----- Reasoning config -----
    reasoning_cfg = {"effort": reasoning_effort} if reasoning_effort else None

    # ----- Text config (object-only at top level) -----
    array_wrapped = False
    items_key = "items"  # we’ll unwrap this key after parsing if we wrapped
    effective_schema = None

    if json_schema:
        # If user gave an array schema, wrap it
        if isinstance(json_schema, dict) and json_schema.get("type") == "array":
            array_wrapped = True
            effective_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    items_key: deepcopy(json_schema)  # the original array schema
                },
                "required": [items_key]
            }
        else:
            effective_schema = json_schema

        text_cfg: Dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": strict_schema,
                "schema": effective_schema,
            },
            "verbosity": text_verbosity,
        }
    else:
        text_cfg = {"format": "plain_text", "verbosity": text_verbosity}

    # ----- Call API (.parse gives you output_parsed when using json_schema) -----
    resp = client.responses.parse(
        model=model,
        input=input_items,
        text=text_cfg,
        reasoning=reasoning_cfg,
        tools=[],
        tool_choice="none",
        store=False,
    )

    text = resp.output_text

    if json_schema:
        _json = json.loads(text)
        if array_wrapped:
            _json = _json.get(items_key, [])
        return (resp, _json) if include_raw else _json

    # Plain text path
    return (resp, text) if include_raw else text


if __name__ == "__main__":
    # Your previous array schema (now supported via auto-wrap):
    example_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "experiment": {"type": "string"},
            },
            "required": ["title", "experiment"]
        }
    }

    data = generate_response(
        model="gpt-5",
        system_prompt="You are a careful assistant. Return only what the schema allows.",
        prompt="Extract a list of studies or experiments in this paper.",
        file_paths=["ZivonyEtEimer2021.pdf"],
        json_schema=example_schema,  # array schema OK; function wraps & unwraps for you
        schema_name="PaperExtraction",
        strict_schema=True,
        reasoning_effort="low",
        text_verbosity="low",
    )
    print(data)  # -> a Python list[dict]
