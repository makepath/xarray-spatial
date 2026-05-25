"""Case-insensitive HTTP(S) scheme routing for SSRF protection (#2332).

Issue #2321 sub-task 5.

Background
----------
Several routing call sites in ``xrspatial/geotiff/`` historically used
``startswith(('http://', 'https://'))`` to decide whether a string source
should be opened by ``_HTTPSource`` (which runs SSRF + DNS-pinning checks
via ``_validate_http_url``) or handed off to fsspec. That comparison is
case-sensitive, so a URL like ``HTTP://169.254.169.254/latest/meta-data``
slipped past ``_HTTPSource`` entirely and fell through to fsspec, which
has no SSRF allow-list. Uppercase schemes are valid per RFC 3986 sect. 3.1
(``scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )``, case-insensitive).

The fix centralizes scheme detection on a single helper, ``_is_http_source``,
that does ``urlparse(url).scheme.lower() in ('http', 'https')``, and routes
every call site through it.

These tests exercise:

* The helper itself across mixed-case schemes.
* ``_open_source`` returning ``_HTTPSource`` for uppercase URLs.
* The dispatch boolean in every other call site (reader, writer, sidecar,
  dask backend, gpu backend, fsspec classifier).
* The SSRF allow-list still rejecting uppercase URLs that resolve to
  private / loopback / link-local addresses.

All tests are offline: ``socket.getaddrinfo`` is monkeypatched so the
validator never opens a real connection.
"""
from __future__ import annotations

import socket

import pytest

from xrspatial.geotiff import UnsafeURLError
from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff import _sources as _sources_mod


# ---------------------------------------------------------------------------
# Helpers (mirrors test_ssrf_hardening_1664.py)
# ---------------------------------------------------------------------------


def _fake_getaddrinfo(ip: str):
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestIsHttpSourceHelper:
    """``_is_http_source`` is the single source of truth for HTTP routing."""

    @pytest.mark.parametrize("url", [
        'http://example.com/x.tif',
        'https://example.com/x.tif',
        'HTTP://example.com/x.tif',
        'HTTPS://example.com/x.tif',
        'Http://example.com/x.tif',
        'hTTpS://example.com/x.tif',
        'http://EXAMPLE.COM/x.tif',  # host case must not matter either
    ])
    def test_http_schemes_match(self, url):
        assert _sources_mod._is_http_source(url) is True

    @pytest.mark.parametrize("url", [
        's3://bucket/key.tif',
        'gs://bucket/key.tif',
        'az://container/blob.tif',
        'abfs://container/blob.tif',
        'file:///etc/passwd',
        'ftp://example.com/x.tif',
        'gopher://example.com/',
        'memory://x.tif',
        '/local/path/file.tif',
        'relative/path.tif',
        'C:\\windows\\file.tif',
    ])
    def test_non_http_schemes_do_not_match(self, url):
        assert _sources_mod._is_http_source(url) is False

    @pytest.mark.parametrize("value", [None, 42, b'http://x', object()])
    def test_non_string_does_not_match(self, value):
        # Be defensive: routing call sites also gate on isinstance(_, str)
        # in some places, but the helper itself must not raise on junk.
        assert _sources_mod._is_http_source(value) is False

    def test_empty_string_does_not_match(self):
        assert _sources_mod._is_http_source('') is False

    def test_scheme_only_prefix_does_not_match(self):
        # ``urlparse('http')`` returns scheme=''; only ``http:`` or
        # ``http://`` should classify as HTTP.
        assert _sources_mod._is_http_source('http') is False

    def test_scheme_colon_no_slashes_classifies_as_http(self):
        # ``urlparse('http:foo').scheme == 'http'``: this is broader than
        # the old ``startswith('http://')`` gate but is RFC-correct. The
        # validator rejects these downstream as "no hostname", so the
        # security posture is unchanged. Locking the broader classifier
        # in here keeps any future tightening explicit. Issue #2332.
        assert _sources_mod._is_http_source('http:foo') is True
        assert _sources_mod._is_http_source('HTTP:foo') is True

    def test_open_source_http_colon_no_hostname_raises(self):
        # End-to-end follow-up: ``_open_source('http:foo')`` now routes
        # into ``_HTTPSource``, which calls ``_validate_http_url`` and
        # raises ``UnsafeURLError('... has no hostname')``. The previous
        # case-sensitive gate would have sent this to fsspec instead.
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source('http:foo')


# ---------------------------------------------------------------------------
# Dispatch: ``_open_source`` must route uppercase URLs through ``_HTTPSource``
# ---------------------------------------------------------------------------


class TestOpenSourceRoutesUppercase:
    """``_open_source('HTTP://...')`` must build an ``_HTTPSource``.

    We intercept ``_HTTPSource.__init__`` so the test never opens a real
    HTTP connection; getting the call at all is what we are verifying.
    """

    def test_uppercase_http_routes_to_http_source(self, monkeypatch):
        calls = []

        def _fake_init(self, url, *args, **kwargs):
            calls.append(url)
            # Skip the real validator / urllib3 pool setup.
            self._url = url

        monkeypatch.setattr(
            _sources_mod._HTTPSource, '__init__', _fake_init)
        src = _sources_mod._open_source('HTTP://example.com/x.tif')
        assert isinstance(src, _sources_mod._HTTPSource)
        assert calls == ['HTTP://example.com/x.tif']

    def test_uppercase_https_routes_to_http_source(self, monkeypatch):
        calls = []

        def _fake_init(self, url, *args, **kwargs):
            calls.append(url)
            self._url = url

        monkeypatch.setattr(
            _sources_mod._HTTPSource, '__init__', _fake_init)
        src = _sources_mod._open_source('HTTPS://example.com/x.tif')
        assert isinstance(src, _sources_mod._HTTPSource)
        assert calls == ['HTTPS://example.com/x.tif']

    def test_mixed_case_routes_to_http_source(self, monkeypatch):
        calls = []

        def _fake_init(self, url, *args, **kwargs):
            calls.append(url)
            self._url = url

        monkeypatch.setattr(
            _sources_mod._HTTPSource, '__init__', _fake_init)
        src = _sources_mod._open_source('hTTpS://example.com/x.tif')
        assert isinstance(src, _sources_mod._HTTPSource)
        assert calls == ['hTTpS://example.com/x.tif']


# ---------------------------------------------------------------------------
# Dispatch booleans elsewhere in the code base
# ---------------------------------------------------------------------------


class TestDispatchBooleansAreCaseInsensitive:
    """Every routing site must use the centralized helper, not startswith.

    Each call site below historically read::

        source.startswith(('http://', 'https://'))

    which is the bug. We assert ``_is_http_source`` returns True for the
    uppercase forms; the implementation modules import and call the same
    helper at the dispatch site.
    """

    @pytest.mark.parametrize("url", [
        'HTTP://example.com/x.tif',
        'HTTPS://example.com/x.tif',
        'Http://example.com/x.tif',
    ])
    def test_helper_recognizes_uppercase(self, url):
        assert _sources_mod._is_http_source(url) is True

    def test_is_fsspec_uri_excludes_uppercase_http(self):
        # ``_is_fsspec_uri`` is the partner classifier in both
        # ``_sources.py`` and ``_writer.py``. If it returned True for
        # ``HTTP://...`` the writer would hand the URL to fsspec instead
        # of raising the typed "writes not supported over HTTP" error.
        assert _sources_mod._is_fsspec_uri('HTTP://example.com/x.tif') is False
        assert _sources_mod._is_fsspec_uri('HTTPS://example.com/x.tif') is False
        # sanity: real fsspec URIs still classify as fsspec
        assert _sources_mod._is_fsspec_uri('s3://b/k.tif') is True

    def test_writer_is_fsspec_uri_excludes_uppercase_http(self):
        from xrspatial.geotiff import _writer as _writer_mod
        assert _writer_mod._is_fsspec_uri('HTTP://example.com/x.tif') is False
        assert _writer_mod._is_fsspec_uri('HTTPS://example.com/x.tif') is False
        assert _writer_mod._is_fsspec_uri('s3://b/k.tif') is True

    def test_sidecar_helper_is_case_insensitive(self):
        from xrspatial.geotiff import _sidecar as _sidecar_mod
        assert _sidecar_mod._is_http_url('HTTP://example.com/x.tif') is True
        assert _sidecar_mod._is_http_url('HTTPS://example.com/x.tif') is True
        assert _sidecar_mod._is_http_url('http://example.com/x.tif') is True
        assert _sidecar_mod._is_http_url('s3://b/k.tif') is False


# ---------------------------------------------------------------------------
# End-to-end: uppercase scheme + private host must still be rejected
# ---------------------------------------------------------------------------


class TestUppercaseSchemeStillRejectsPrivateHosts:
    """The whole point of the fix: uppercase URLs go through the SSRF gate.

    Before the fix, ``HTTP://169.254.169.254/...`` would skip the validator
    and try to open via fsspec. After the fix, it routes through
    ``_HTTPSource``, which calls ``_validate_http_url``, which raises
    ``UnsafeURLError``.
    """

    @pytest.mark.parametrize("scheme", ['HTTP', 'HTTPS', 'Http', 'hTTpS'])
    @pytest.mark.parametrize("ip", [
        '127.0.0.1',
        '169.254.169.254',
        '10.0.0.1',
        '192.168.1.1',
        '0.0.0.0',
    ])
    def test_private_host_rejected_regardless_of_scheme_case(
            self, monkeypatch, scheme, ip):
        monkeypatch.setattr(socket, 'getaddrinfo', _fake_getaddrinfo(ip))
        url = f'{scheme}://attacker.test/x.tif'
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(url)

    @pytest.mark.parametrize("scheme", ['HTTP', 'HTTPS', 'Http'])
    def test_localhost_rejected_regardless_of_scheme_case(
            self, monkeypatch, scheme):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(f'{scheme}://localhost:8080/x.tif')

    @pytest.mark.parametrize("scheme", ['HTTP', 'HTTPS', 'Http'])
    def test_uppercase_scheme_to_127_literal_rejected(
            self, monkeypatch, scheme):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(f'{scheme}://127.0.0.1/x.tif')

    def test_open_source_uppercase_private_host_raises(self, monkeypatch):
        """End-to-end: ``_open_source`` -> ``_HTTPSource`` -> validator.

        Confirms the dispatch wiring actually drives the URL through the
        validator (not just that the validator works in isolation).
        """
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo('169.254.169.254'))
        # Make sure the env override is not set; the validator skips
        # resolution when ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS`` is on.
        monkeypatch.delenv(
            'XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', raising=False)
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(
                'HTTP://metadata.google.internal/computeMetadata/v1/')


# ---------------------------------------------------------------------------
# Writer: HTTP(S) destinations must raise a typed error, not a raw OSError
# ---------------------------------------------------------------------------


class TestWriterRejectsHttpTargets:
    """``_write_bytes(_, 'HTTP://...')`` must raise ``NotImplementedError``.

    Without the early gate the uppercase URL fell through ``_is_fsspec_uri``
    (correctly returns False) and into the local file write path, which
    surfaced an OS-specific ``OSError`` for the colon-in-filename. The
    typed error matches the lowercase-HTTP behaviour and points users at
    the supported destinations. Follow-up to issue #2332 review.
    """

    @pytest.mark.parametrize("url", [
        'http://example.com/x.tif',
        'https://example.com/x.tif',
        'HTTP://example.com/x.tif',
        'HTTPS://example.com/x.tif',
        'Http://example.com/x.tif',
    ])
    def test_write_bytes_rejects_http(self, url):
        from xrspatial.geotiff import _writer as _writer_mod
        with pytest.raises(NotImplementedError) as excinfo:
            _writer_mod._write_bytes(b'IIxxxx', url)
        msg = str(excinfo.value)
        assert 'HTTP' in msg
        assert url in msg
