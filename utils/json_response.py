"""Parse model JSON while preserving accidentally escaped command characters."""

import json


def parse_model_json(value: str) -> dict:
    for _ in range(100):
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("Model response must be a JSON object")
            return parsed
        except json.JSONDecodeError as exc:
            if exc.msg != "Invalid \\escape":
                raise
            # Apostrophes need no escaping in JSON, but models often escape
            # the quotes around an action argument. Other invalid escapes are
            # preserved as literal backslashes (for example a regex \d).
            if value[exc.pos + 1:exc.pos + 2] == "'":
                value = value[:exc.pos] + value[exc.pos + 1:]
            else:
                value = value[:exc.pos] + "\\" + value[exc.pos:]
    raise ValueError("Too many invalid JSON escapes in model response")
