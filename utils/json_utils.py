import json
import re

def extract_json(raw_text: str) -> dict:
    """Strip markdown fences / stray text and parse the model's JSON output safely."""
    text = raw_text.strip()

    # Remove markdown code fences if the model added them anyway
    text = re.sub(r"^```json\s*|```$", "", text, flags=re.MULTILINE).strip()

    # If there's leading/trailing junk, isolate the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {e}\nRaw output:\n{raw_text}")


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)