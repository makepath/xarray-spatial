"""SSRF defenses on ``_HTTPSource`` (issue #1664).

Before #1664, ``open_geotiff(url=...)`` accepted any URL: ``file://``,
``http://127.0.0.1:6379/``, ``http://169.254.169.254/...`` (cloud
metadata). It also had no explicit timeouts and no explicit redirect
cap.

These tests cover the validator in isolation -- they do NOT make real
HTTP calls. ``socket.getaddrinfo`` is monkeypatched per-test to control
what the validator sees.
"""
from __future__ import annotations

import socket

import pytest

from xrspatial.geotiff import UnsafeURLError
from xrspatial.geotiff import _reader as _reader_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_getaddrinfo(ip: str):
    """Return a getaddrinfo replacement that always resolves to *ip*.

    Mirrors the real return tuple layout: each element is
    ``(family, type, proto, canonname, sockaddr)``. The validator only
    looks at index 4 (sockaddr) so the rest is filler.
    """
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# Scheme allow-list
# ---------------------------------------------------------------------------


class TestSchemeAllowList:
    def test_https_accepted(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
        # example.com -> a public IP -- should pass.
        _reader_mod._validate_http_url('https://example.com/cog.tif')

    def test_http_accepted(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
        _reader_mod._validate_http_url('http://example.com/cog.tif')

    def test_file_scheme_rejected(self):
        with pytest.raises(UnsafeURLError) as excinfo:
            _reader_mod._validate_http_url('file:///etc/passwd')
        msg = str(excinfo.value).lower()
        assert "scheme" in msg
        assert "'file'" in str(excinfo.value).lower()

    def test_gopher_scheme_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('gopher://example.com/')

    def test_ftp_scheme_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('ftp://example.com/x.tif')

    def test_empty_url_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('')

    def test_non_string_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(None)  # type: ignore[arg-type]

    def test_env_var_widens_allow_list(self, monkeypatch):
        monkeypatch.setenv(
            'XRSPATIAL_GEOTIFF_ALLOWED_SCHEMES', 'ftp,gopher')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
        # Now ftp:// should be accepted (host still validated).
        _reader_mod._validate_http_url('ftp://example.com/x.tif')


# ---------------------------------------------------------------------------
# Host filtering -- private / loopback / link-local
# ---------------------------------------------------------------------------


class TestPrivateHostBlocking:
    @pytest.mark.parametrize("ip", [
        '127.0.0.1',
        '127.0.0.5',
        '10.0.0.1',
        '172.16.5.5',
        '192.168.1.1',
        '169.254.169.254',  # cloud metadata
        '0.0.0.0',
    ])
    def test_ipv4_private_rejected(self, monkeypatch, ip):
        monkeypatch.setattr(socket, 'getaddrinfo', _fake_getaddrinfo(ip))
        with pytest.raises(UnsafeURLError) as excinfo:
            _reader_mod._validate_http_url('http://attacker.test/x.tif')
        msg = str(excinfo.value).lower()
        assert "private" in msg or "loopback" in msg or "link-local" in msg

    @pytest.mark.parametrize("ip", [
        '::1',          # IPv6 loopback
        'fe80::1',      # IPv6 link-local
        'fc00::1',      # IPv6 unique-local
    ])
    def test_ipv6_private_rejected(self, monkeypatch, ip):
        monkeypatch.setattr(socket, 'getaddrinfo', _fake_getaddrinfo(ip))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://attacker.test/x.tif')

    def test_localhost_literal_rejected(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://localhost:8080/x.tif')

    def test_public_ip_accepted(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('8.8.8.8'))
        _reader_mod._validate_http_url('http://example.com/x.tif')

    def test_env_override_allows_private(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        # No exception expected.
        _reader_mod._validate_http_url('http://127.0.0.1:8080/cog.tif')

    def test_env_override_truthy_values(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        for v in ('1', 'true', 'TRUE', 'yes', 'on'):
            monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', v)
            _reader_mod._validate_http_url('http://127.0.0.1/x.tif')

    def test_dns_rebind_partial_private_rejected(self, monkeypatch):
        """If ANY resolved IP is private, the URL is rejected.

        This blocks DNS-rebinding tricks where a hostile DNS server
        returns both a public and a private IP for the same name.
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

    def test_unresolvable_host_rejected(self, monkeypatch):
        def _broken(host, port, *args, **kwargs):
            raise socket.gaierror(-2, 'Name or service not known')
        monkeypatch.setattr(socket, 'getaddrinfo', _broken)
        with pytest.raises(UnsafeURLError) as excinfo:
            _reader_mod._validate_http_url('http://nope.example.invalid/x.tif')
        assert "resolve" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------


class TestHTTPTimeouts:
    def test_default_connect_timeout(self, monkeypatch):
        monkeypatch.delenv(
            'XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT', raising=False)
        assert _reader_mod._http_connect_timeout() == 10.0

    def test_default_read_timeout(self, monkeypatch):
        monkeypatch.delenv(
            'XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT', raising=False)
        assert _reader_mod._http_read_timeout() == 30.0

    def test_env_override_connect(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT', '2.5')
        assert _reader_mod._http_connect_timeout() == 2.5

    def test_env_override_read(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT', '7')
        assert _reader_mod._http_read_timeout() == 7.0

    def test_env_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv(
            'XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT', 'not-a-float')
        assert _reader_mod._http_connect_timeout() == 10.0

    def test_env_zero_falls_back(self, monkeypatch):
        # Zero is not a useful timeout; treat as missing.
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT', '0')
        assert _reader_mod._http_read_timeout() == 30.0


# ---------------------------------------------------------------------------
# Redirect cap
# ---------------------------------------------------------------------------


def test_redirect_cap_is_set():
    """The module-level constant is what the urllib3 pool gets."""
    assert _reader_mod._HTTP_MAX_REDIRECTS == 5


# ---------------------------------------------------------------------------
# Integration: _HTTPSource.__init__ runs the validator
# ---------------------------------------------------------------------------


class TestHTTPSourceConstructor:
    def test_file_url_rejected_at_init(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._HTTPSource('file:///etc/passwd')

    def test_localhost_url_rejected_at_init(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._HTTPSource('http://127.0.0.1:6379/probe.tif')

    def test_metadata_url_rejected_at_init(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('169.254.169.254'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._HTTPSource(
                'http://169.254.169.254/latest/meta-data/')

    def test_escape_hatch_allows_localhost(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        src = _reader_mod._HTTPSource('http://127.0.0.1:8080/cog.tif')
        # Timeout values are pulled from the env-aware helpers.
        assert src._connect_timeout == 10.0
        assert src._read_timeout == 30.0

    def test_unsafe_url_error_is_value_error(self):
        """``UnsafeURLError`` is a ``ValueError`` so existing handlers work."""
        with pytest.raises(ValueError):
            _reader_mod._HTTPSource('file:///etc/passwd')

    def test_unsafe_url_error_carries_url(self):
        try:
            _reader_mod._HTTPSource('file:///etc/passwd')
        except UnsafeURLError as e:
            assert e.url == 'file:///etc/passwd'
        else:
            pytest.fail("UnsafeURLError not raised")


# ---------------------------------------------------------------------------
# read_to_array dispatcher honours the SSRF check
# ---------------------------------------------------------------------------


def test_read_to_array_rejects_file_url():
    """The top-level dispatcher refuses file:// URLs via _HTTPSource."""
    # ``file://`` does not match the http(s) prefix in _open_source, so
    # it does NOT hit _HTTPSource at all -- it gets routed via the path
    # branch which interprets the URL literally and fails to find the
    # file. The relevant guarantee is just: arbitrary local file access
    # via ``file://`` URL does not succeed quietly.
    from xrspatial.geotiff._reader import read_to_array
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        read_to_array('file:///etc/passwd')


def test_open_source_rejects_loopback_http(monkeypatch):
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
    with pytest.raises(UnsafeURLError):
        _reader_mod._open_source('http://127.0.0.1:8080/x.tif')
