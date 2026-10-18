"""Explicit optional model pricing shared by single-task and batch runners."""

import math


def add_pricing_arguments(parser):
    parser.add_argument("--input-usd-per-million", type=float,
                        help="Input USD per million tokens; provide both rates or neither")
    parser.add_argument("--output-usd-per-million", type=float,
                        help="Output USD per million tokens; omitted rates leave cost null")


def validate_prices(input_rate, output_rate):
    if (input_rate is None) != (output_rate is None):
        raise ValueError("Provide both --input-usd-per-million and --output-usd-per-million, or neither")
    for rate in (input_rate, output_rate):
        if rate is not None and (not math.isfinite(rate) or rate < 0):
            raise ValueError("Token prices must be finite and nonnegative")


def estimate_cost(usage, input_rate, output_rate):
    validate_prices(input_rate, output_rate)
    if input_rate is None or usage["source"] == "unavailable":
        return None
    return round((usage["total"]["input_tokens"] * input_rate
                  + usage["total"]["output_tokens"] * output_rate) / 1_000_000, 6)
