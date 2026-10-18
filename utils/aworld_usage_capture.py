"""Collect provider-reported token usage from the AWorld OpenAI adapter."""

from contextlib import contextmanager
from contextvars import ContextVar

from utils.llm_usage import usage_from_result


@contextmanager
def capture_aworld_usage(on_usage=None):
    # AWorld calls this module-level function after each provider completion,
    # but its Runners.run result may not expose the same usage to our agents.
    import aworld.models.openai_provider as provider_module
    from aworld.runner import Runners

    original = provider_module.usage_process
    original_run = Runners.run
    role = ContextVar("itbench_usage_role", default="unattributed")
    calls = []

    async def run_with_role(*args, **kwargs):
        agent = kwargs.get("agent")
        name = getattr(agent, "name", "unattributed")
        if callable(name):
            name = name()
        token = role.set(str(name).lower().replace(" agent", "").replace(" ", "_"))
        try:
            return await original_run(*args, **kwargs)
        finally:
            role.reset(token)

    def capture(usage, *args, **kwargs):
        normalized = usage_from_result(type("UsageResult", (), {"usage": usage})())
        if normalized["total_tokens"]:
            normalized["role"] = role.get()
            calls.append(normalized)
            if on_usage is not None:
                on_usage(normalized)
        return original(usage, *args, **kwargs)

    provider_module.usage_process = capture
    Runners.run = staticmethod(run_with_role)
    try:
        yield calls
    finally:
        provider_module.usage_process = original
        Runners.run = staticmethod(original_run)
