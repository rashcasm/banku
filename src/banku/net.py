"""One shared HTTP client for every API call.

Two measured problems this solves, both of which made Banku feel slow for
reasons that had nothing to do with the model:

1. `httpx.post(...)` builds a new client per call, so every question paid a
   fresh DNS + TCP + TLS handshake. On this connection that was ~3.3 s for the
   first Groq request and ~0.4 s once the socket was reused.

2. This network advertises an IPv6 route but cannot actually use it. Python
   tries the AAAA address first, stalls for about two seconds, then falls back
   to IPv4. `curl -4` finished in 0.34 s where the default took 3.3 s. Binding
   the local address to 0.0.0.0 forces IPv4 and removes the stall.

The client is kept alive between calls. If the server drops an idle connection,
the request is retried once on a fresh socket rather than failing the reply.
"""

from __future__ import annotations

import httpx

_client: httpx.Client | None = None
_client_timeout: float | None = None


def _make(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        transport=httpx.HTTPTransport(
            local_address="0.0.0.0",  # IPv4 only; see module docstring
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=600),
        ),
    )


def get_client(timeout: float = 30.0) -> httpx.Client:
    global _client, _client_timeout
    if _client is None or _client_timeout != timeout:
        if _client is not None:
            _client.close()
        _client = _make(timeout)
        _client_timeout = timeout
    return _client


def _reset() -> None:
    global _client, _client_timeout
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001 - already broken
            pass
    _client = None
    _client_timeout = None


def post(url: str, *, timeout: float = 30.0, **kwargs: object) -> httpx.Response:
    """POST with a warm IPv4 connection, retrying once if it went stale."""
    try:
        return get_client(timeout).post(url, **kwargs)  # type: ignore[arg-type]
    except (httpx.TransportError, httpx.RemoteProtocolError):
        _reset()
        return get_client(timeout).post(url, **kwargs)  # type: ignore[arg-type]
