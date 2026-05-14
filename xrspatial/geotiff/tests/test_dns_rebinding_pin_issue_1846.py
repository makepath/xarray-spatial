"""DNS-rebinding TOCTOU defence on ``_HTTPSource`` (issue #1846).

Before this issue, ``_validate_http_url`` resolved the hostname once at
construction (and again on each redirect), but urllib3 resolved the
hostname a *second* time at connect time. A hostile DNS server could
return a public IP to the validator and a private IP to the connect
call (classic DNS rebinding). The fix pins the validated IP into the
TCP connection while keeping the original hostname in the Host header
and TLS SNI.

These tests confirm:

1. Rebound DNS does not reach the private IP: the TCP socket goes to
   the validated public IP regardless of what ``getaddrinfo`` returns
   later.
2. ``_validate_http_url`` returns the pinned IP (so callers can wire
   it through to the connection).
3. Redirects re-resolve-and-re-pin per hop, so a redirect to a new
   hostname is freshly validated.
"""
from __future__ import annotations

import socket

import pytest

from xrspatial.geotiff import UnsafeURLError
from xrspatial.geotiff import _reader as _reader_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ip_resolver(ip: str):
    """Return a ``getaddrinfo`` replacement that resolves any host to *ip*."""
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


def _switching_resolver(ips: list[str]):
    """Return a resolver that yields a different IP on each call.

    After the script is exhausted, it sticks on the final IP. This is
    how we simulate a rebinding DNS server: the first call (validation)
    returns ``ips[0]``, the second call (would-be TCP connect) returns
    ``ips[1]``.
    """
    state = {'i': 0}

    def _resolver(host, port, *args, **kwargs):
        idx = min(state['i'], len(ips) - 1)
        state['i'] += 1
        ip = ips[idx]
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# _validate_http_url returns the pinned IP
# ---------------------------------------------------------------------------


class TestValidatorReturnsPinnedIP:
    def test_returns_first_public_ip(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('93.184.216.34'))
        ip = _reader_mod._validate_http_url('https://example.com/cog.tif')
        assert ip == '93.184.216.34'

    def test_returns_first_public_ipv6(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('2606:2800:220:1::1'))
        ip = _reader_mod._validate_http_url('https://example.com/cog.tif')
        assert ip == '2606:2800:220:1::1'

    def test_returns_none_when_escape_hatch_enabled(self, monkeypatch):
        """With the escape hatch we skip resolution and skip pinning.

        Callers that opt into private hosts knowingly accept the looser
        guarantee, and we don't want to force them to provide a literal
        IP. Returning ``None`` tells ``_HTTPSource`` to fall back to
        urllib3's default DNS path.
        """
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('127.0.0.1'))
        ip = _reader_mod._validate_http_url('http://127.0.0.1:8080/cog.tif')
        assert ip is None

    def test_raises_when_any_ip_private(self, monkeypatch):
        """Existing SSRF guarantee is preserved.

        If any resolved IP is private we still raise; we don't silently
        pick a public one and pin to that.
        """
        def _resolver(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 ('8.8.8.8', port or 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 ('127.0.0.1', port or 0)),
            ]
        monkeypatch.setattr(socket, 'getaddrinfo', _resolver)
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://attacker.test/x.tif')


# ---------------------------------------------------------------------------
# DNS rebinding defeated: the actual TCP connect targets the pinned IP
# ---------------------------------------------------------------------------


class TestPinnedConnectionTarget:
    def test_init_records_pinned_ip(self, monkeypatch):
        pytest.importorskip("urllib3")
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')
        assert src._pinned_ip == '93.184.216.34'

    def test_rebind_does_not_reach_private_ip(self, monkeypatch):
        """End-to-end rebinding test.

        Validator sees ``93.184.216.34`` (safe public IP). After that,
        any further DNS resolution returns ``127.0.0.1`` (rebound to
        loopback). The TCP connection must still target the validated
        public IP.

        We intercept ``socket.create_connection`` to record the target
        address rather than actually opening a socket. ``read_range``
        will fail when the mock returns no data; we only care that the
        connection was attempted against the pinned IP.
        """
        pytest.importorskip("urllib3")

        # First getaddrinfo call (validation) returns public IP. Every
        # subsequent call returns the rebound private IP.
        monkeypatch.setattr(
            socket, 'getaddrinfo',
            _switching_resolver(['93.184.216.34', '127.0.0.1']))

        src = _reader_mod._HTTPSource('https://example.com/cog.tif')
        assert src._pinned_ip == '93.184.216.34'

        # Capture every TCP target the pool tries to dial. We don't
        # actually want to open sockets, so we raise after recording.
        attempted: list[tuple] = []

        class _StopConnect(OSError):
            pass

        def _fake_create_connection(address, *args, **kwargs):
            attempted.append(address)
            raise _StopConnect("intercepted in test")

        # urllib3's HTTPConnection uses ``urllib3.util.connection.
        # create_connection`` indirectly via _new_conn. Our pinned
        # connection overrides _new_conn to call ``socket.create_
        # connection`` directly, so patching ``socket.create_connection``
        # is sufficient.
        monkeypatch.setattr(
            socket, 'create_connection', _fake_create_connection)

        # Calling read_range goes through the pinned pool. The mocked
        # create_connection raises before any real network I/O. urllib3
        # wraps OSError in NewConnectionError / MaxRetryError; either
        # way an exception bubbles up.
        with pytest.raises(Exception):
            src.read_range(0, 100)

        # The validated public IP was used as the TCP target, not the
        # rebound private one.
        assert attempted, "expected at least one TCP connect attempt"
        target_ip = attempted[0][0]
        assert target_ip == '93.184.216.34', (
            f"TCP connect went to {target_ip!r}, expected pinned "
            f"public IP 93.184.216.34. DNS rebinding succeeded.")
        assert target_ip != '127.0.0.1'

    def test_host_header_and_sni_preserved(self, monkeypatch):
        """Host header (and TLS SNI for HTTPS) stay set to the original
        hostname, not the IP literal. Required for HTTP virtual hosting
        and TLS certificate verification.
        """
        pytest.importorskip("urllib3")
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')

        # Build the pinned pool the same way _request would, and
        # inspect a fresh connection's attributes.
        pool = src._get_pinned_pool(
            'https', 'example.com', 443, '93.184.216.34')
        conn = pool._new_conn()
        try:
            # ``host`` is what becomes the Host header (urllib3 uses
            # ``self.host`` when building HTTP requests).
            assert conn.host == 'example.com'
            # The connection class carries the pinned IP. ``_new_conn``
            # dials this directly via ``socket.create_connection`` and
            # never consults DNS again.
            assert conn.pinned_ip == '93.184.216.34'
            # ``server_hostname`` is the TLS SNI value the pool feeds
            # into the connection at handshake time; must be the
            # original hostname so cert verification works. The pool
            # stashes pool-construction extras (anything not in the
            # explicit kwarg list) in ``conn_kw``, which it splats into
            # the connection constructor in ``_new_conn``.
            assert conn.server_hostname == 'example.com'
            # The freshly-built connection has not yet hit the wire,
            # so cert validation hasn't run; we're checking the
            # *configuration* that will be used.
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Redirects: each hop is independently validated and pinned
# ---------------------------------------------------------------------------


class _MockPoolResponse:
    def __init__(self, status: int, location: str | None = None,
                 data: bytes = b''):
        self.status = status
        self.headers = {'Location': location} if location else {}
        self.data = data


class _MockPool:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append(url)
        assert kwargs.get('redirect') is False
        if self.script:
            return self.script.pop(0)
        return _MockPoolResponse(200, data=b'OK')


class TestRedirectRevalidates:
    def test_redirect_to_safe_host_revalidates(self, monkeypatch):
        """A redirect from safe-host -> also-safe re-runs validation on
        the new hostname and pins the new IP.
        """
        pytest.importorskip("urllib3")
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://safe-host.example.com/a.tif')
        assert src._pinned_ip == '93.184.216.34'

        # Record every host the validator sees by hooking ``getaddrinfo``.
        # The hop-by-hop redirect loop in ``_request`` calls
        # ``_validate_http_url`` on each ``Location``, which in turn
        # calls ``getaddrinfo`` once per host. The new hop is on a
        # different hostname so it resolves to a different public IP
        # (we script the resolver to switch on the second call).
        seen_hosts: list[str] = []

        def _tracking_resolver(host, port, *args, **kwargs):
            seen_hosts.append(host)
            # First host (safe-host.example.com) -> 93.184.216.34.
            # Second host (also-safe.example.com) -> 1.1.1.1.
            ip = '1.1.1.1' if 'also-safe' in host else '93.184.216.34'
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0))]
        monkeypatch.setattr(socket, 'getaddrinfo', _tracking_resolver)

        # Use a scripted mock pool so we don't touch the network. We
        # override _pool_for_request so the redirect loop receives our
        # mock for both hops.
        mock_pool = _MockPool([
            _MockPoolResponse(
                302,
                location='https://also-safe.example.com/b.tif'),
            _MockPoolResponse(200, data=b'ok'),
        ])

        def _stub_pool_for_request(url, pinned_ip):
            return mock_pool

        monkeypatch.setattr(
            src, '_pool_for_request', _stub_pool_for_request)

        data = src.read_range(0, 10)
        assert data == b'ok'

        # The Location host was resolved (i.e. re-validated). The
        # initial host might not appear here because the constructor
        # ran *before* the tracking resolver was installed.
        assert any('also-safe.example.com' == h for h in seen_hosts), (
            f"Redirect target was not re-validated. Hosts seen by "
            f"getaddrinfo during the request: {seen_hosts!r}")
        # Both URLs were issued through the mock, confirming the loop
        # walked both hops.
        assert mock_pool.calls == [
            'https://safe-host.example.com/a.tif',
            'https://also-safe.example.com/b.tif',
        ]

    def test_redirect_to_private_still_rejected(self, monkeypatch):
        """Pinning doesn't weaken the existing redirect-to-private guard."""
        pytest.importorskip("urllib3")
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')

        # Now the redirect target resolves to loopback.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver('127.0.0.1'))

        # Use the existing _pool slot for mocking (matches the rest of
        # the SSRF tests in this codebase).
        from xrspatial.geotiff.tests.test_ssrf_hardening_1664 import (
            _MockPool as _SsrfMockPool,
            _MockPoolResponse as _SsrfResp,
        )
        src._pool = _SsrfMockPool([
            _SsrfResp(302, location='http://attacker.test/inner.tif'),
        ])
        with pytest.raises(UnsafeURLError):
            src.read_range(0, 100)
