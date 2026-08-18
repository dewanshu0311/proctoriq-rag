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

DEFAULT_MODEL = "llama-3.1-8b-instant"
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
    max_tokens: int = 400
    api_keys: list[str] = field(default_factory=list)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    sleep = staticmethod(time.sleep)

    _clients: list[Any] = field(default_factory=list, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)
    calls: int = field(default=0, init=False)
    rotations: int = field(default=0, init=False)

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

            self._clients = [Groq(api_key=key) for key in self.api_keys]
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
                return (completion.choices[0].message.content or "").strip()
            except Exception as error:  # noqa: BLE001 - SDK raises many shapes
                last_error = error
                if not _is_retryable(error):
                    raise
                if len(clients) > 1:
                    self._index += 1
                    self.rotations += 1
                    continue
                self.sleep(self.retry.delay_for(attempt))

        raise GroqError(
            f"Groq request failed after {self.retry.max_attempts} attempts: {last_error}"
        ) from last_error


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
