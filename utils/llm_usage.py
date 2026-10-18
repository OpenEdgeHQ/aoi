"""Normalize token usage returned by AWorld runner calls."""


def usage_from_result(result) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        read = usage.get
    else:
        read = lambda key, default=0: getattr(usage, key, default)
    input_tokens = read("prompt_tokens", 0) or read("input_tokens", 0) or 0
    output_tokens = read("completion_tokens", 0) or read("output_tokens", 0) or 0
    total_tokens = read("total_tokens", 0) or input_tokens + output_tokens
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }
