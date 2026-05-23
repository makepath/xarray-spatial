"""Uppercase URL schemes must not dodge SSRF hardening (issue #2323).

URL schemes are case-insensitive per RFC 3986. Before this fix, the
geotiff reader's dispatch helpers compared schemes case-sensitively, so
a URL like ``HTTP://127.0.0.1/foo.tif`` would skip ``_HTTPSource`` (and
its SSRF allow-list + pinned DNS) and land on the fsspec branch via
``_is_fsspec_uri``. Tests below pin each dispatch site to the
case-insensitive behaviour.

No real HTTP calls are made: ``socket.getaddrinfo`` is monkeypatched per
test.
"""
from __future__ import annotations

import socket

import pytest

from xrspatial.geotiff import UnsafeURLError
from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff import _sources as _sources_mod
from xrspatial.geotiff._sidecar import _is_http_url as _sidecar_is_http_url
from xrspatial.geotiff._writer import _is_fsspec_uri as _writer_is_fsspec_uri


# Unique tmp-name prefix to keep parallel-rockout temp paths apart.
_ISSUE = "2323"


def _fake_getaddrinfo(ip: str):
    """getaddrinfo replacement that always resolves to *ip*."""
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# _is_http_url / _is_fsspec_uri helpers
# ---------------------------------------------------------------------------


class TestIsHttpUrlCaseInsensitive:
    @pytest.mark.parametrize("url", [
        "http://example.com/x.tif",
        "https://example.com/x.tif",
        "HTTP://example.com/x.tif",
        "HTTPS://example.com/x.tif",
        "Http://example.com/x.tif",
        "hTTpS://example.com/x.tif",
        "HTTP://127.0.0.1/foo.tif",
    ])
    def test_http_variants_recognised(self, url):
        assert _sources_mod._is_http_url(url) is True
        assert _reader_mod._is_http_url(url) is True
        assert _sidecar_is_http_url(url) is True

    @pytest.mark.parametrize("path", [
        "s3://bucket/x.tif",
        "S3://bucket/x.tif",
        "gs://bucket/x.tif",
        "memory://buf",
        "/local/path.tif",
        "relative/path.tif",
        "",
    ])
    def test_non_http_rejected(self, path):
        assert _sources_mod._is_http_url(path) is False

    def test_non_string_rejected(self):
        assert _sources_mod._is_http_url(None) is False
        assert _sources_mod._is_http_url(b"http://x/y") is False
        assert _sources_mod._is_http_url(123) is False


class TestIsFsspecUriExcludesHttpCaseInsensitive:
    @pytest.mark.parametrize("url", [
        "HTTP://127.0.0.1/foo.tif",
        "HTTPS://example.com/x.tif",
        "Http://example.com/x.tif",
        "hTTpS://example.com/x.tif",
        "http://example.com/x.tif",
    ])
    def test_uppercase_http_not_fsspec(self, url):
        # The bug: before the fix, uppercase URLs slipped past the
        # http/https exclusion in _is_fsspec_uri and were routed to the
        # fsspec branch (bypassing SSRF defences).
        assert _sources_mod._is_fsspec_uri(url) is False
        assert _writer_is_fsspec_uri(url) is False

    @pytest.mark.parametrize("uri", [
        "s3://bucket/x.tif",
        "S3://bucket/x.tif",  # fsspec accepts case-insensitive schemes too
        "gs://bucket/x.tif",
        "memory://buffer",
    ])
    def test_real_fsspec_uris_still_match(self, uri):
        assert _sources_mod._is_fsspec_uri(uri) is True


# ---------------------------------------------------------------------------
# _open_source dispatch -- uppercase URL must hit _HTTPSource
# (and therefore SSRF validation), not _CloudSource
# ---------------------------------------------------------------------------


class TestOpenSourceUppercaseDispatch:
    def test_uppercase_loopback_rejected(self, monkeypatch):
        # 127.0.0.1 is in the private/loopback range; the SSRF validator
        # in _HTTPSource must reject it. If the URL were dispatched to
        # fsspec instead, this would either silently succeed against a
        # localhost service or raise a different error.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(f'HTTP://127.0.0.1/x_{_ISSUE}.tif')

    def test_uppercase_https_loopback_rejected(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(
                f'HTTPS://localhost/x_{_ISSUE}.tif')

    def test_uppercase_metadata_ip_rejected(self, monkeypatch):
        # 169.254.169.254 is the cloud-metadata service IP that SSRF
        # attacks typically target. The validator treats link-local as
        # private and must reject it whether the scheme is upper or
        # lower case.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('169.254.169.254'))
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(
                f'HTTP://metadata.example/x_{_ISSUE}.tif')

    def test_uppercase_public_routes_to_http_source(self, monkeypatch):
        # A public IP should construct _HTTPSource successfully (rather
        # than silently going to fsspec). We don't make a real request:
        # the pinned-DNS resolution is enough to prove the dispatch
        # branch picked _HTTPSource.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
        src = _sources_mod._open_source(
            f'HTTP://example.com/x_{_ISSUE}.tif')
        try:
            assert type(src).__name__ == '_HTTPSource'
        finally:
            src.close()


# ---------------------------------------------------------------------------
# read_to_array dispatcher -- uppercase URL must take the COG-HTTP path
# (which validates via _HTTPSource), not the fsspec _CloudSource path
# ---------------------------------------------------------------------------


class TestReadToArrayUppercaseDispatch:
    def test_uppercase_url_takes_http_path(self, monkeypatch):
        """``read_to_array`` must route uppercase URLs through the
        ``_read_cog_http`` path so the SSRF allow-list runs.

        We stub ``_read_cog_http`` to capture the dispatch decision
        without making any network calls.
        """
        captured = {}

        def _fake_read_cog_http(source, **kwargs):
            captured['source'] = source
            captured['kwargs'] = kwargs
            # Mimic the real return shape just enough to satisfy the
            # caller: it isn't inspected here because we raise to short
            # circuit any downstream logic.
            raise RuntimeError("stubbed _read_cog_http reached")

        monkeypatch.setattr(
            _reader_mod, '_read_cog_http', _fake_read_cog_http)

        with pytest.raises(RuntimeError, match="stubbed _read_cog_http"):
            _reader_mod.read_to_array(
                f'HTTP://example.com/x_{_ISSUE}.tif')
        assert captured.get('source') == (
            f'HTTP://example.com/x_{_ISSUE}.tif'
        )

    def test_lowercase_url_still_takes_http_path(self, monkeypatch):
        # Regression guard: don't break the existing lowercase path.
        captured = {}

        def _fake_read_cog_http(source, **kwargs):
            captured['source'] = source
            raise RuntimeError("stubbed _read_cog_http reached")

        monkeypatch.setattr(
            _reader_mod, '_read_cog_http', _fake_read_cog_http)

        with pytest.raises(RuntimeError):
            _reader_mod.read_to_array(
                f'http://example.com/x_{_ISSUE}.tif')
        assert captured['source'] == (
            f'http://example.com/x_{_ISSUE}.tif'
        )
