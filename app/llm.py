"""Pluggable LLM interface with a deterministic offline fallback.

Agents never talk to a model provider directly; they depend on the small
``LLMClient`` protocol below. Two implementations ship with the project:

* ``OfflineLLM`` — the default. Fully deterministic and template-driven:
  it renders a narrative from the structured ``context`` dict the caller
  supplies. No network, no API key, identical output for identical input,
  which keeps demos and tests reproducible.
* ``OpenAILLM`` — optional. Used only when ``MERCHANT_LLM_PROVIDER=openai``
  and an ``OPENAI_API_KEY`` is present. The ``openai`` package is imported
  lazily so it is not a hard dependency.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .config import settings


class LLMClient(Protocol):
    """Minimal text-completion interface used by the agents."""

    def complete(
        self,
        prompt: str,
        system: str = "",
        context: Optional[dict] = None,
    ) -> str:
        ...


class OfflineLLM:
    """Deterministic, template-based stand-in for an LLM.

    The Report Writer agent passes its structured findings in ``context``;
    this class renders them into a readable narrative. Because the output
    is a pure function of the input, pipeline results are reproducible in
    tests and demos.
    """

    name = "offline"

    def complete(
        self,
        prompt: str,
        system: str = "",
        context: Optional[dict] = None,
    ) -> str:
        ctx = context or {}
        if ctx.get("kind") == "merchant_report":
            return self._render_merchant_report(ctx)
        # Generic deterministic fallback: echo a bounded digest of the
        # prompt so behaviour is predictable for any caller.
        digest = " ".join(prompt.split())[:280]
        return f"[offline-llm] {digest}"

    @staticmethod
    def _render_merchant_report(ctx: dict) -> str:
        lines = [
            f"Merchant intelligence report for {ctx['merchant_name']} "
            f"({ctx['category']}, {ctx['region']}).",
            (
                f"Overall risk is assessed as {ctx['risk_level']} "
                f"(score {ctx['risk_score']}/100)."
            ),
        ]
        profile_bits = []
        if ctx.get("transaction_count") is not None:
            profile_bits.append(
                f"{ctx['transaction_count']} transactions totalling "
                f"${ctx['total_volume']:,.2f}"
            )
        if ctx.get("chargeback_rate") is not None:
            profile_bits.append(
                f"a chargeback rate of {ctx['chargeback_rate']:.2%}"
            )
        if ctx.get("volume_trend_pct") is not None:
            direction = "grew" if ctx["volume_trend_pct"] >= 0 else "declined"
            profile_bits.append(
                f"volume {direction} by {abs(ctx['volume_trend_pct']):.1f}% "
                "across the observation window"
            )
        if profile_bits:
            lines.append("Activity summary: " + "; ".join(profile_bits) + ".")
        for flag in ctx.get("risk_flags", []):
            lines.append(f"Risk flag: {flag}.")
        for insight in ctx.get("insights", []):
            lines.append(f"Insight: {insight}.")
        if ctx.get("recommendation"):
            lines.append(f"Recommendation: {ctx['recommendation']}.")
        return "\n".join(lines)


class OpenAILLM:
    """Thin wrapper over the OpenAI chat API (optional dependency)."""

    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "MERCHANT_LLM_PROVIDER=openai requires the 'openai' package"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(
        self,
        prompt: str,
        system: str = "",
        context: Optional[dict] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.2
        )
        return resp.choices[0].message.content or ""


def get_llm(provider: Optional[str] = None) -> LLMClient:
    """Factory honouring MERCHANT_LLM_PROVIDER (default: offline)."""
    provider = (provider or settings.llm_provider).lower()
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "MERCHANT_LLM_PROVIDER=openai but OPENAI_API_KEY is not set"
            )
        return OpenAILLM(settings.openai_api_key, settings.openai_model)
    if provider != "offline":
        raise ValueError(f"unknown LLM provider: {provider!r}")
    return OfflineLLM()
