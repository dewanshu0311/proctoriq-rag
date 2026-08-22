"""Groq client with retry, backoff, and optional key rotation.

Single key is the default. Fifty questions is fifty calls, comfortably inside one
free-tier key's limits, and rotation is complexity carrying its own failure modes
— a rotating client that silently masks a bad key is worse than one that fails
loudly. Rotation exists as an opt-in fallback (``GROQ_ROTATE_KEYS=1``) for if we
ever actually hit a limit, and is not exercised otherwise.

No key is needed for extractive mode. Nothing here is imported unless generative
mode is selected.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# llama-3.1-8b-instant was REMOVED from Groq mid-competition (404
# model_not_found), after it had already produced the probe 6a arm. See D-041.
#
# gpt-oss models are reasoning models: they spend completion tokens on an
# internal reasoning trace before emitting content. At max_tokens=300 the trace
# consumed the whole budget and `content` came back EMPTY — which the router
# would have read as an unparseable reply and silently fallen back on. Hence the
# larger default.
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_ROTATION_KEYS = 16


@runtime_checkable
class ChatClient(Protocol):
    """Structural type so tests can inject a stub without touching the network."""

    def complete(self, prompt: str, **kwargs: Any) -> str: ...


class GroqError(RuntimeError):
    """Raised when every retry and every available key has been exhausted."""


def load_dotenv(path: Path | str = ".env") -> dict[str, str]:
    """Minimal ``.env`` reader. Does not overwrite existing environment values."""
    path = Path(path)
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if value:
            values[name] = value
            os.environ.setdefault(name, value)
    return values


def discover_keys(rotate: bool | None = None) -> list[str]:
    """Return the API keys to use, primary first.

    On Kaggle the key comes from ``UserSecretsClient`` rather than the
    environment, so this returning empty is not an error until a call is made.
    """
    load_dotenv()
    if rotate is None:
        rotate = os.environ.get("GROQ_ROTATE_KEYS", "0").strip() in {"1", "true", "yes"}

    primary = os.environ.get("GROQ_API_KEY", "").strip()
    keys = [primary] if primary else []

    # Fall back to the numbered secrets when the unsuffixed one is absent.
    # This is not rotation — it is resolution. A missing GROQ_API_KEY once caused
    # a notebook run to silently produce the WRONG ARM: the router never fired,
    # refusals no-op'd, and it wrote a valid-looking 50-row submission that was
    # actually the extractive baseline. Single key stays the default; this only
    # stops a naming mismatch from being mistaken for "no key available".
    if not keys:
        for index in range(1, MAX_ROTATION_KEYS + 1):
            value = os.environ.get(f"GROQ_API_KEY_{index}", "").strip()
            if value:
                keys.append(value)
                break

    if rotate:
        for index in range(1, MAX_ROTATION_KEYS + 1):
            value = os.environ.get(f"GROQ_API_KEY_{index}", "").strip()
            if value and value not in keys:
                keys.append(value)

    return keys


@dataclass
class RetryPolicy:
    """Exponential backoff with jitter. Bounded so a run cannot hang forever."""

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        raw = min(self.base_delay * (2 ** attempt), self.max_delay)
        return raw * (1.0 + random.uniform(0.0, self.jitter))


@dataclass
class GroqChatClient:
    """Thin wrapper over the Groq SDK.

    Temperature defaults to 0. That is not a style preference: the probe design
    requires answers to be stable across submissions, and while only extractive
    mode is byte-identical by construction, a non-zero temperature would make
    generative runs gratuitously irreproducible on top of that.
    """

    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 2500
    api_keys: list[str] = field(default_factory=list)
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    #: Per-request timeout, seconds. WITHOUT THIS A HUNG REQUEST BLOCKS FOREVER.
    #: Observed: a sweep sat for 26 minutes on one call with 22s of CPU across 31
    #: minutes of wall time — blocked on a socket, with the retry/backoff logic
    #: never reached because nothing ever raised. Retries only help if something
    #: fails; a hang is not a failure.
    request_timeout: float = 60.0
    sleep = staticmethod(time.sleep)

    _clients: list[Any] = field(default_factory=list, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)
    calls: int = field(default=0, init=False)
    rotations: int = field(default=0, init=False)

    #: Set on every completion so callers can distinguish a TRUNCATED reply from
    #: a failed one. Reasoning models spend completion tokens on an internal trace
    #: before emitting content, so an exhausted budget returns finish_reason
    #: "length" with EMPTY content — which is indistinguishable from an API
    #: failure at the text layer, and silently degrades to the fallback answer.
    last_finish_reason: str = field(default="", init=False)
    last_completion_tokens: int = field(default=0, init=False)
    truncations: int = field(default=0, init=False)
    max_completion_tokens_seen: int = field(default=0, init=False)

    #: Keys that returned a daily-quota 429. Retired for the rest of the run — a
    #: TPD limit does not clear in seconds, so retrying an exhausted key just
    #: burns wall time. Measured: key 1 hit 199,637 of 200,000 TPD while keys 2-9
    #: still had full budget, so they are separate quotas and rotation is the
    #: right response rather than backoff.
    exhausted: set = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if not self.api_keys:
            self.api_keys = discover_keys()

    @property
    def clients(self) -> list[Any]:
        if not self._clients:
            if not self.api_keys:
                raise GroqError(
                    "No Groq API key available. Set GROQ_API_KEY in .env locally, or "
                    "provide it via Kaggle Secrets. Extractive mode needs no key."
                )
            from groq import Groq

            self._clients = [
                Groq(api_key=key, timeout=self.request_timeout, max_retries=0)
                for key in self.api_keys
            ]
        return self._clients

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """One completion, retrying on transient failures.

        On a rate-limit error the next key is tried immediately when rotation is
        enabled; otherwise it backs off and retries the same key.
        """
        clients = self.clients
        last_error: Exception | None = None

        for attempt in range(self.retry.max_attempts):
            client = clients[self._index % len(clients)]
            try:
                completion = client.chat.completions.create(
                    model=kwargs.get("model", self.model),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                self.calls += 1
                choice = completion.choices[0]
                self.last_finish_reason = choice.finish_reason or ""
                usage = getattr(completion, "usage", None)
                self.last_completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.max_completion_tokens_seen = max(
                    self.max_completion_tokens_seen, self.last_completion_tokens
                )
                text = (choice.message.content or "").strip()
                if self.last_finish_reason == "length" and not text:
                    self.truncations += 1
                return text
            except Exception as error:  # noqa: BLE001 - SDK raises many shapes
                last_error = error
                if not _is_retryable(error):
                    raise

                if _is_daily_quota(error):
                    self.exhausted.add(self._index % len(clients))
                    if len(self.exhausted) >= len(clients):
                        raise GroqError(
                            f"All {len(clients)} Groq keys have hit their daily token "
                            f"limit. Last: {error}"
                        ) from error

                if len(clients) > 1:
                    # Skip past any key already known to be out of budget.
                    for _ in range(len(clients)):
                        self._index += 1
                        self.rotations += 1
                        if (self._index % len(clients)) not in self.exhausted:
                            break
                    continue
                self.sleep(self.retry.delay_for(attempt))

        raise GroqError(
            f"Groq request failed after {self.retry.max_attempts} attempts: {last_error}"
        ) from last_error


def _is_daily_quota(error: Exception) -> bool:
    """A tokens-per-day 429, as distinct from a per-minute one.

    A TPD limit does not clear in seconds. Backing off against it stalls the run:
    a sweep sat for 26 minutes because the SDK's own retries honoured a ~55s
    Retry-After INSIDE this loop's retries, nesting the sleeps. Rotate instead.
    """
    text = str(error).lower()
    return "429" in text and ("per day" in text or "tpd" in text)


def _is_retryable(error: Exception) -> bool:
    """Rate limits, timeouts and 5xx are worth retrying; bad requests are not."""
    name = type(error).__name__.lower()
    text = str(error).lower()
    if any(token in name for token in ("ratelimit", "timeout", "connection", "apistatus")):
        return True
    return any(
        token in text
        for token in ("rate limit", "429", "timeout", "temporarily", "503", "502", "500",
                      "overloaded", "try again")
    )
