"""urllib3 is the only HTTP transport for ``_HTTPSource`` (issue #2050).

Before #2050, ``_HTTPSource.read_range`` and ``_HTTPSource.read_all`` had
two code paths: a urllib3 path that pinned the TCP connection to the IP
returned by ``_validate_http_url``, and a stdlib ``urllib.request``
fallback that re-resolved the hostname at request time. With urllib3
missing from ``install_requires``, a production install could land on
the stdlib path and silently lose the DNS-rebinding IP pin from #1846.

#2050 makes ``urllib3`` a hard install dependency and removes the stdlib
fallback. These tests lock in that contract.
"""
from __future__ import annotations

import inspect
import socket

import pytest

from xrspatial.geotiff import _reader as _reader_mod

# ---------------------------------------------------------------------------
# urllib3 is a hard runtime dependency
# ---------------------------------------------------------------------------


def test_urllib3_is_importable():
    """urllib3 is in install_requires; importing the module must work."""
    import urllib3  # noqa: F401


def test_reader_imports_urllib3_at_module_level():
    """The reader keeps a module-level reference to urllib3.

    Module-level rather than deferred-import makes it impossible to ship
    a build where urllib3 is missing and the code silently degrades to
    a different transport.
    """
    assert hasattr(_reader_mod, 'urllib3')


def test_get_http_pool_returns_a_pool_manager():
    """``_get_http_pool`` is no longer allowed to return None.

    Pre-#2050 it returned ``None`` when urllib3 was missing, which is
    what routed callers into the stdlib fallback.
    """
    import urllib3
    pool = _reader_mod._get_http_pool()
    assert pool is not None
    assert isinstance(pool, urllib3.PoolManager)


# ---------------------------------------------------------------------------
# The stdlib fallback symbols are gone
# ---------------------------------------------------------------------------


def test_stdlib_opener_helper_is_removed():
    """``_get_stdlib_opener`` was the entry point for the unpinned path."""
    assert not hasattr(_reader_mod, '_get_stdlib_opener')
    assert not hasattr(_reader_mod, '_stdlib_opener')


def test_validating_redirect_handler_is_removed():
    """The stdlib redirect handler is gone with the stdlib transport."""
    assert not hasattr(_reader_mod, '_ValidatingRedirectHandler')


def test_reader_does_not_import_urllib_request():
    """``urllib.request`` is no longer needed once the stdlib path is gone.

    A residual ``import urllib.request`` at module scope would be a
    smell -- the only legitimate consumer was the deleted opener.
    """
    src = inspect.getsource(_reader_mod)
    # The token has to appear in *executable* form, not just inside a
    # comment or docstring. Strip comment lines and check the rest.
    code_lines = [
        line for line in src.splitlines()
        if not line.lstrip().startswith('#')
    ]
    code = '\n'.join(code_lines)
    assert 'import urllib.request' not in code, (
        "urllib.request should not be imported now that the stdlib "
        "HTTP fallback is removed (#2050)."
    )


def test_read_range_source_has_no_stdlib_branch():
    """``read_range`` body must not reference ``urllib.request``."""
    src = inspect.getsource(_reader_mod._HTTPSource.read_range)
    assert 'urllib.request' not in src
    assert 'stdlib_opener' not in src


def test_read_all_source_has_no_stdlib_branch():
    """``read_all`` body must not reference ``urllib.request``."""
    src = inspect.getsource(_reader_mod._HTTPSource.read_all)
    assert 'urllib.request' not in src
    assert 'stdlib_opener' not in src


# ---------------------------------------------------------------------------
# urllib3 path still works -- mock the pool and exercise read_range / read_all
# ---------------------------------------------------------------------------


def _fake_getaddrinfo(ip: str):
    def _resolver(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


class _MockResp:
    def __init__(self, status, data=b'', content_range=None,
                 content_length=None):
        self.status = status
        self.data = data
        self._body = data
        self.headers = {}
        if content_range is not None:
            self.headers['Content-Range'] = content_range
        # ``read_range`` (post #2264) does a Content-Length preflight; let
        # callers either pin it explicitly or default to len(data).
        if content_length is None and data:
            self.headers['Content-Length'] = str(len(data))
        elif content_length is not None:
            self.headers['Content-Length'] = str(content_length)

    def stream(self, amt=65536, decode_content=True):
        # Yield the body in a single chunk; ``_read_capped`` reads
        # whatever ``stream()`` produces.
        if self._body:
            yield self._body

    def release_conn(self):
        pass


class _MockPool:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._resp


def test_read_range_uses_urllib3_pool(monkeypatch):
    """Sanity check: a successful 206 round-trips through ``_request``."""
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
    src = _reader_mod._HTTPSource('https://example.com/cog.tif')
    body = b'A' * 100
    pool = _MockPool(_MockResp(
        status=206, data=body, content_range='bytes 0-99/1000'))
    src._pool = pool

    data = src.read_range(0, 100)
    assert data == body
    assert len(pool.calls) == 1
    method, url, kwargs = pool.calls[0]
    assert method == 'GET'
    assert kwargs.get('redirect') is False
    assert kwargs.get('headers', {}).get('Range') == 'bytes=0-99'
    # Post #2264: the GET must request a streaming body so the cap is
    # enforced on the wire rather than after urllib3 has already
    # buffered ``resp.data``.
    assert kwargs.get('preload_content') is False


def test_read_all_uses_urllib3_pool(monkeypatch):
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
    src = _reader_mod._HTTPSource('https://example.com/cog.tif')
    body = b'tiff-bytes'
    pool = _MockPool(_MockResp(status=200, data=body))
    src._pool = pool

    data = src.read_all()
    assert data == body
    assert len(pool.calls) == 1


def test_read_range_short_circuits_zero_length(monkeypatch):
    """No HTTP traffic for length<=0 -- behaviour preserved from pre-#2050."""
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo('93.184.216.34'))
    src = _reader_mod._HTTPSource('https://example.com/cog.tif')
    pool = _MockPool(_MockResp(status=206, data=b''))
    src._pool = pool

    assert src.read_range(0, 0) == b''
    assert src.read_range(10, -5) == b''
    assert pool.calls == []


# ---------------------------------------------------------------------------
# install_requires advertises urllib3
# ---------------------------------------------------------------------------


def test_install_requires_lists_urllib3():
    """setup.cfg must list urllib3 so deployed installs get it.

    Without this, a wheel built today would let pip resolve the install
    without urllib3, and the import at top of _reader would fail at
    open_geotiff() time rather than at install time.
    """
    import pathlib
    setup_cfg = (
        pathlib.Path(_reader_mod.__file__).resolve()
        .parent.parent.parent / 'setup.cfg'
    )
    if not setup_cfg.exists():  # pragma: no cover
        pytest.skip(f"setup.cfg not found at {setup_cfg}")
    text = setup_cfg.read_text()
    # Locate the install_requires block and confirm urllib3 appears in it.
    head, _, tail = text.partition('install_requires =')
    assert tail, "install_requires section not found in setup.cfg"
    # The block ends at the next top-level key (lines that start in
    # column 0). Walk until we see one.
    block_lines = []
    for line in tail.splitlines()[1:]:
        if line and not line.startswith((' ', '\t')):
            break
        block_lines.append(line.strip())
    assert 'urllib3' in block_lines, (
        f"urllib3 must be in install_requires; found: {block_lines}"
    )
