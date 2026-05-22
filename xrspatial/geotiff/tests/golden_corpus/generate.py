"""Deterministic generator for the geotiff golden corpus.

Reads ``manifest.yaml`` and rebuilds every fixture under the corpus
directory. The generator is the contract for what a fixture is: anything
not expressible here belongs in the manifest schema, not in ad-hoc code.

Determinism guarantees:

* fixtures are iterated in declared order, files are emitted in sorted-id
  order, and no timestamps are written;
* pixel data is produced from ``pixel_pattern`` (ramp / checker / noise /
  uniform) with a per-fixture ``pixel_seed`` when randomness is needed;
* file modification times are normalised to a fixed epoch after writing so
  re-runs produce byte-identical sidecar metadata.

Usage::

    python -m xrspatial.geotiff.tests.golden_corpus.generate
    python -m xrspatial.geotiff.tests.golden_corpus.generate --dry-run
    python -m xrspatial.geotiff.tests.golden_corpus.generate --only <id>

Dry-run validates the manifest and reports what would be written without
touching disk. The smoke test in this PR uses dry-run; Phase 2 PRs will
flip to real writes once each fixture group lands.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

# rasterio and pyyaml are not part of the package's install_requires.
# They are pulled in by the test extra and by typical dev environments.
# Import errors are surfaced with a clear hint rather than a bare
# ModuleNotFoundError so a contributor running the script outside the
# test environment gets actionable output.
try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PyYAML is required to read the golden corpus manifest. "
        "Install with `pip install pyyaml` or use the test extras."
    ) from exc

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.yaml"
DEFAULT_OUTPUT_DIR = HERE / "fixtures"

# Fixed epoch used to normalise on-disk mtimes (2020-01-01 UTC). Picked
# arbitrarily; the only requirement is that it is constant across runs.
DETERMINISTIC_EPOCH = 1577836800

REQUIRED_FIELDS = (
    "id",
    "description",
    "width",
    "height",
    "bands",
    "dtype",
    "byte_order",
    "layout",
    "planar_config",
    "compression",
    "predictor",
    "photometric",
    "pixel_pattern",
)

ALLOWED_BYTE_ORDER = {"little", "big"}
ALLOWED_LAYOUT = {"stripped", "tiled"}
ALLOWED_PLANAR = {"contig", "separate"}
ALLOWED_COMPRESSION = {
    "none", "deflate", "lzw", "lerc", "jpeg", "packbits", "zstd",
}
ALLOWED_PHOTOMETRIC = {"minisblack", "miniswhite", "rgb", "ycbcr"}
ALLOWED_PATTERN = {"ramp", "checker", "noise", "uniform", "noise_with_corners"}
ALLOWED_PREDICTOR = {1, 2, 3}


class ManifestError(ValueError):
    """Raised when the manifest fails validation."""


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and return the parsed manifest. No validation here."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ManifestError(f"{path} did not parse to a mapping")
    return data


def _merge_defaults(defaults: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Return defaults merged with entry. Entry keys win."""
    merged = dict(defaults)
    merged.update(entry)
    return merged


def _validate_one(entry: dict[str, Any], seen_ids: set[str]) -> None:
    """Raise ManifestError if `entry` is malformed."""
    missing = [k for k in REQUIRED_FIELDS if k not in entry]
    if missing:
        raise ManifestError(
            f"fixture {entry.get('id', '?')!r} missing required fields: "
            f"{sorted(missing)}"
        )

    fid = entry["id"]
    if not isinstance(fid, str) or not fid:
        raise ManifestError(f"fixture id must be a non-empty string, got {fid!r}")
    if fid in seen_ids:
        raise ManifestError(f"duplicate fixture id: {fid!r}")
    seen_ids.add(fid)

    if entry["byte_order"] not in ALLOWED_BYTE_ORDER:
        raise ManifestError(
            f"{fid}: byte_order must be one of {sorted(ALLOWED_BYTE_ORDER)}"
        )
    if entry["layout"] not in ALLOWED_LAYOUT:
        raise ManifestError(
            f"{fid}: layout must be one of {sorted(ALLOWED_LAYOUT)}"
        )
    if entry["layout"] == "tiled":
        ts = entry.get("tile_size")
        if not isinstance(ts, int) or ts <= 0 or ts % 16 != 0:
            raise ManifestError(
                f"{fid}: tiled layout requires tile_size as a positive int "
                f"multiple of 16, got {ts!r}"
            )
    else:
        bs = entry.get("blocksize")
        if not isinstance(bs, int) or bs <= 0:
            raise ManifestError(
                f"{fid}: stripped layout requires blocksize as a positive int, "
                f"got {bs!r}"
            )

    if entry["planar_config"] not in ALLOWED_PLANAR:
        raise ManifestError(
            f"{fid}: planar_config must be one of {sorted(ALLOWED_PLANAR)}"
        )
    if entry["compression"] not in ALLOWED_COMPRESSION:
        raise ManifestError(
            f"{fid}: compression must be one of {sorted(ALLOWED_COMPRESSION)}"
        )
    if entry["predictor"] not in ALLOWED_PREDICTOR:
        raise ManifestError(
            f"{fid}: predictor must be one of {sorted(ALLOWED_PREDICTOR)}"
        )
    if entry["photometric"] not in ALLOWED_PHOTOMETRIC:
        raise ManifestError(
            f"{fid}: photometric must be one of {sorted(ALLOWED_PHOTOMETRIC)}"
        )
    if entry["pixel_pattern"] not in ALLOWED_PATTERN:
        raise ManifestError(
            f"{fid}: pixel_pattern must be one of {sorted(ALLOWED_PATTERN)}"
        )

    # noise_with_corners needs at least 2x2 so the four corners are
    # distinct pixels; otherwise the corner stamping silently collapses.
    if entry["pixel_pattern"] == "noise_with_corners":
        if entry.get("width", 0) < 2 or entry.get("height", 0) < 2:
            raise ManifestError(
                f"{fid}: pixel_pattern 'noise_with_corners' requires "
                f"width >= 2 and height >= 2, got "
                f"{entry.get('width')}x{entry.get('height')}"
            )

    # dtype must be a recognised numpy dtype.
    try:
        np.dtype(entry["dtype"])
    except TypeError as exc:
        raise ManifestError(f"{fid}: dtype {entry['dtype']!r} not a numpy dtype") from exc

    bands = entry["bands"]
    if not isinstance(bands, int) or bands < 1:
        raise ManifestError(f"{fid}: bands must be a positive int, got {bands!r}")

    for k in ("width", "height"):
        v = entry[k]
        if not isinstance(v, int) or v <= 0:
            raise ManifestError(f"{fid}: {k} must be a positive int, got {v!r}")

    # nodata: null, number, "nan", or "miniswhite"
    nd = entry.get("nodata", None)
    if nd is not None and not isinstance(nd, (int, float)) and nd not in (
        "nan", "miniswhite",
    ):
        raise ManifestError(
            f"{fid}: nodata must be null, a number, \"nan\", or \"miniswhite\"; "
            f"got {nd!r}"
        )

    # crs is null or exactly one of epsg / wkt / citation
    crs = entry.get("crs", None)
    if crs is not None:
        if not isinstance(crs, dict):
            raise ManifestError(f"{fid}: crs must be null or a mapping, got {crs!r}")
        keys = set(crs)
        if keys not in ({"epsg"}, {"wkt"}, {"citation"}):
            raise ManifestError(
                f"{fid}: crs map must have exactly one of "
                f"{{epsg, wkt, citation}}, got {sorted(keys)}"
            )

    transform = entry.get("transform")
    if transform is not None:
        if not (isinstance(transform, list) and len(transform) == 6):
            raise ManifestError(
                f"{fid}: transform must be a 6-element list, got {transform!r}"
            )

    overviews = entry.get("overviews")
    if overviews:
        if not isinstance(overviews, list) or not all(
            isinstance(x, int) and x > 1 for x in overviews
        ):
            raise ManifestError(
                f"{fid}: overviews must be a list of ints > 1, got {overviews!r}"
            )

    # predictor 3 is the floating-point predictor; predictor 2 is the
    # horizontal differencing predictor for integer data. Catching the
    # wrong pairing here gives a clear error instead of a confusing one
    # from rasterio at write time.
    pred = entry["predictor"]
    dtype_kind = np.dtype(entry["dtype"]).kind
    if pred == 3 and dtype_kind != "f":
        raise ManifestError(
            f"{fid}: predictor 3 (floating-point) requires a float dtype, "
            f"got dtype {entry['dtype']!r}"
        )
    if pred == 2 and dtype_kind == "f":
        raise ManifestError(
            f"{fid}: predictor 2 (horizontal) requires an integer dtype, "
            f"got dtype {entry['dtype']!r}"
        )

    if "external_overview" in entry and not isinstance(
        entry["external_overview"], bool
    ):
        raise ManifestError(
            f"{fid}: external_overview must be a bool, "
            f"got {entry['external_overview']!r}"
        )

    gdal_md = entry.get("gdal_metadata")
    if gdal_md is not None:
        if not isinstance(gdal_md, dict):
            raise ManifestError(
                f"{fid}: gdal_metadata must be a mapping of "
                f"{{domain: {{item: value}}}}, got {gdal_md!r}"
            )
        for domain, items in gdal_md.items():
            if not isinstance(domain, str):
                raise ManifestError(
                    f"{fid}: gdal_metadata domain keys must be strings, "
                    f"got {domain!r}"
                )
            if not isinstance(items, dict):
                raise ManifestError(
                    f"{fid}: gdal_metadata[{domain!r}] must be a mapping, "
                    f"got {items!r}"
                )

    extra_tags = entry.get("extra_tags")
    if extra_tags is not None:
        if not isinstance(extra_tags, dict):
            raise ManifestError(
                f"{fid}: extra_tags must be a mapping, got {extra_tags!r}"
            )
        for k in extra_tags:
            # bool is a subclass of int in Python, so the (str, int) check
            # would otherwise let `True` / `False` through as tag code 1/0.
            if isinstance(k, bool) or not isinstance(k, (str, int)):
                raise ManifestError(
                    f"{fid}: extra_tags keys must be strings or ints, "
                    f"got {k!r}"
                )

    if "cog" in entry and not isinstance(entry["cog"], bool):
        raise ManifestError(
            f"{fid}: cog must be a bool, got {entry['cog']!r}"
        )
    if entry.get("cog") and entry.get("external_overview"):
        raise ManifestError(
            f"{fid}: cog=true is incompatible with external_overview=true; "
            f"the COG spec requires internal overviews"
        )

    if "sparse" in entry and not isinstance(entry["sparse"], bool):
        raise ManifestError(
            f"{fid}: sparse must be a bool, got {entry['sparse']!r}"
        )
    if entry.get("sparse"):
        # GDAL itself honours SPARSE_OK on stripped writers too, but the
        # corpus generator only wires the tiled path (where sparse
        # encoding actually matters for cloud-optimised reads) and the
        # corpus oracle does not yet pin the stripped sparse case.
        # Reject the combination up front so the manifest cannot promise
        # behaviour the generator does not exercise.
        if entry["layout"] != "tiled":
            raise ManifestError(
                f"{fid}: sparse=true is only wired for layout=tiled in "
                f"the corpus generator; GDAL itself accepts SPARSE_OK on "
                f"stripped writers but no corpus fixture exercises that"
            )
        if entry.get("cog"):
            raise ManifestError(
                f"{fid}: sparse=true is incompatible with cog=true; the "
                f"COG copy driver rewrites the file and drops sparse tiles"
            )


def validate(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the parsed manifest and return resolved fixture entries.

    Each returned dict has defaults already merged in.
    """
    version = manifest.get("version")
    if version != 1:
        raise ManifestError(f"unsupported manifest version: {version!r} (expected 1)")

    defaults = manifest.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ManifestError("defaults must be a mapping if present")

    raw_fixtures = manifest.get("fixtures") or []
    if not isinstance(raw_fixtures, list):
        raise ManifestError("fixtures must be a list")

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_fixtures:
        if not isinstance(raw, dict):
            raise ManifestError(f"fixture entry must be a mapping, got {raw!r}")
        merged = _merge_defaults(defaults, raw)
        _validate_one(merged, seen)
        resolved.append(merged)

    return resolved


def _make_pixels(entry: dict[str, Any]) -> np.ndarray:
    """Build deterministic pixel data for a fixture entry.

    Returns an array of shape (bands, height, width) in the requested dtype.
    """
    bands = entry["bands"]
    h = entry["height"]
    w = entry["width"]
    dtype = np.dtype(entry["dtype"])
    pattern = entry["pixel_pattern"]
    n = bands * h * w

    if pattern == "ramp":
        flat = np.arange(n, dtype=np.float64)
    elif pattern == "checker":
        rows = np.arange(h)[:, None]
        cols = np.arange(w)[None, :]
        single = ((rows // 8 + cols // 8) % 2).astype(np.float64)
        flat = np.broadcast_to(single, (bands, h, w)).reshape(-1).astype(np.float64)
    elif pattern in ("noise", "noise_with_corners"):
        rng = np.random.default_rng(int(entry.get("pixel_seed", 0)))
        flat = rng.random(n)
    elif pattern == "uniform":
        flat = np.full(n, float(entry.get("pixel_value", 0)), dtype=np.float64)
    else:  # pragma: no cover - validate() rejects unknown patterns
        raise ManifestError(f"unknown pixel_pattern: {pattern!r}")

    if dtype.kind in ("i", "u"):
        info = np.iinfo(dtype)
        # Map [0, 1) for noise / [0, n) for ramp into the dtype range.
        if pattern in ("ramp", "checker", "uniform"):
            arr = flat % (info.max - info.min + 1) + info.min
        else:  # noise / noise_with_corners
            arr = flat * (info.max - info.min) + info.min
        arr = arr.astype(dtype)
    else:
        arr = flat.astype(dtype)

    arr = arr.reshape(bands, h, w)

    # ``noise_with_corners`` plants the dtype's min and max sentinels in the
    # four corner pixels of every band so dtype-edge handling gets exercised
    # by the corpus. Floats keep noise as-is; a NaN sentinel is a separate
    # property tracked by Phase 2 PR 6 (nodata).
    if pattern == "noise_with_corners" and dtype.kind in ("i", "u"):
        info = np.iinfo(dtype)
        lo = dtype.type(info.min)
        hi = dtype.type(info.max)
        arr[:, 0, 0] = lo
        arr[:, 0, -1] = hi
        arr[:, -1, 0] = hi
        arr[:, -1, -1] = lo

    _stamp_nodata_pixels(arr, entry)
    return arr


def _stamp_nodata_pixels(arr: np.ndarray, entry: dict[str, Any]) -> None:
    """Plant a few sentinel pixels at deterministic positions.

    The corpus nodata fixtures (#1930, Phase 2 PR 6) need the oracle to
    exercise nodata-masking semantics, not just the tag round-trip.
    Noise / ramp / uniform patterns are vanishingly unlikely to hit the
    sentinel value on their own for wide integer dtypes (a 16x16 uint16
    raster sees each value with probability 1/65536 per cell), so we
    stamp a small set of cells in-place after pattern generation.

    The cells (top-left, centre, bottom-right) are fixed so re-runs stay
    byte-stable. We stamp only when ``nodata`` resolves to an actual
    sentinel value:

    * a numeric sentinel for integer / float rasters
    * NaN for float rasters with ``nodata: "nan"``
    * the dtype max for ``nodata: "miniswhite"`` (white-as-min)
    """
    nd = entry.get("nodata")
    if nd is None:
        return
    dtype = arr.dtype
    # ``bool`` is a subclass of ``int``; reject it explicitly so a
    # ``nodata: true`` manifest entry can't slip a 1 into the raster.
    # The write-side gate is #1990; this is the matching read-side gate.
    if isinstance(nd, bool):
        return
    if isinstance(nd, (int, float)):
        sentinel: Any = nd
    elif nd == "nan":
        if dtype.kind != "f":
            return
        sentinel = np.nan
    elif nd == "miniswhite":
        if dtype.kind not in ("i", "u"):
            return
        sentinel = np.iinfo(dtype).max
    else:  # pragma: no cover - validate() rejects other shapes
        return
    h = arr.shape[-2]
    w = arr.shape[-1]
    positions = ((0, 0), (h // 2, w // 2), (h - 1, w - 1))
    for b in range(arr.shape[0]):
        for r, c in positions:
            arr[b, r, c] = sentinel


def _resolve_crs(crs_spec: dict[str, Any] | None):
    """Convert a manifest CRS spec into a rasterio CRS or None."""
    if crs_spec is None:
        return None
    from rasterio.crs import CRS

    if "epsg" in crs_spec:
        return CRS.from_epsg(int(crs_spec["epsg"]))
    if "wkt" in crs_spec:
        return CRS.from_wkt(str(crs_spec["wkt"]))
    if "citation" in crs_spec:
        # Citation-only: a WKT keyed only by name, no AUTHORITY tag and
        # no numeric projection parameters. Exercises the oracle's
        # non-EPSG WKT fallback (Phase 2 PR 8 of #1930). PROJ does not
        # resolve this to an EPSG code; on round-trip libgeotiff mutates
        # the WKT (axis order, UNIT AUTHORITY) but preserves to_dict().
        return CRS.from_wkt(
            f'GEOGCS["{crs_spec["citation"]}",DATUM["unknown",SPHEROID["unknown",6378137,0]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
    raise ManifestError(f"unreachable: unknown crs spec {crs_spec!r}")


def _rasterio_kwargs(entry: dict[str, Any]) -> dict[str, Any]:
    """Translate a resolved fixture entry into rasterio open() kwargs.

    Pulled out so the validator-only path can introspect it in dry-run.
    """
    from rasterio.transform import Affine

    kwargs: dict[str, Any] = {
        "driver": "GTiff",
        "width": entry["width"],
        "height": entry["height"],
        "count": entry["bands"],
        "dtype": entry["dtype"],
        "photometric": entry["photometric"].upper(),
        "interleave": (
            "band" if entry["planar_config"] == "separate" else "pixel"
        ),
    }
    if entry["layout"] == "tiled":
        kwargs["tiled"] = True
        kwargs["blockxsize"] = entry["tile_size"]
        kwargs["blockysize"] = entry["tile_size"]
    else:
        kwargs["tiled"] = False
        kwargs["blockysize"] = entry["blocksize"]

    if entry["compression"] != "none":
        kwargs["compress"] = entry["compression"]
        if entry["predictor"] != 1 and entry["compression"] in (
            "deflate", "lzw", "zstd",
        ):
            kwargs["predictor"] = entry["predictor"]
        # Codec level (deflate / zstd / lerc). GDAL accepts -1 / None to
        # mean "default", which we represent by simply not forwarding.
        level = entry.get("compression_level")
        if isinstance(level, int) and level >= 0:
            if entry["compression"] == "deflate":
                kwargs["zlevel"] = level
            elif entry["compression"] == "zstd":
                kwargs["zstd_level"] = level
            elif entry["compression"] == "jpeg":
                kwargs["jpeg_quality"] = level
        if entry["compression"] == "lerc":
            max_z = entry.get("max_z_error")
            if max_z is not None:
                kwargs["max_z_error"] = float(max_z)

    if entry["byte_order"] == "big":
        # GDAL GTiff driver ENDIANNESS creation option. rasterio forwards
        # unknown uppercase kwargs to GDAL as creation options verbatim;
        # the lowercase ``endian`` kwarg is intercepted by rasterio and
        # silently dropped, so we route through the GDAL name directly.
        kwargs["ENDIANNESS"] = "BIG"

    if entry.get("sparse"):
        # GDAL GTiff driver creation option SPARSE_OK=TRUE. Lets the writer
        # elide tiles whose pixels are all zero (or all nodata); their
        # ``TileByteCounts`` entry becomes 0 and the reader treats them as
        # implicit zeros on read. The flag passes through rasterio as a
        # GDAL creation option because rasterio has no native handle for
        # it. See https://gdal.org/drivers/raster/gtiff.html#creation-options
        kwargs["SPARSE_OK"] = "TRUE"

    nd = entry.get("nodata")
    if isinstance(nd, (int, float)):
        kwargs["nodata"] = nd
    elif nd == "nan":
        kwargs["nodata"] = float("nan")
    # "miniswhite" / None: no nodata tag written.

    transform = entry.get("transform")
    if transform is not None:
        kwargs["transform"] = Affine(*transform)

    crs = _resolve_crs(entry.get("crs"))
    if crs is not None:
        kwargs["crs"] = crs

    return kwargs


# Well-known TIFF tag codes for the string keys we accept in extra_tags.
# Keys are stored in lower-case for case-insensitive lookup. The values are
# the TIFF tag numbers from the baseline TIFF 6.0 spec.
_WELL_KNOWN_TIFF_TAGS = {
    "imagedescription": 270,
    "software": 305,
    "artist": 315,
    "copyright": 33432,
    "datetime": 306,
    "documentname": 269,
    "hostcomputer": 316,
}

# Tag codes the tifffile post-pass must preserve when rewriting a file to
# attach extra TIFF tags. These cover:
# * GeoTIFF georeferencing tags (33550, 33922, 34735, 34736, 34737) -- rasterio
#   emits these to encode CRS / transform.
# * 42112 GDAL_METADATA -- holds any cross-domain GDAL metadata rasterio
#   wrote via update_tags(ns=...). Preserved so a fixture can carry both
#   gdal_metadata and extra_tags without losing the GDAL XML.
# tifffile re-derives SampleFormat / ExtraSamples from the pixel dtype on
# write, so those are intentionally not in this list.
_GEOTIFF_TAG_CODES = (33550, 33922, 34735, 34736, 34737, 42112)


def _normalize_extra_tag_key(key: Any) -> tuple[int, str]:
    """Resolve an extra_tags key to a (numeric_code, display_name).

    String keys are matched against `_WELL_KNOWN_TIFF_TAGS` case-insensitively;
    int keys pass through as private tag codes. Unknown string keys raise so
    typos surface at generation time rather than producing a silently
    different file.
    """
    if isinstance(key, int):
        return key, str(key)
    if isinstance(key, str):
        code = _WELL_KNOWN_TIFF_TAGS.get(key.lower())
        if code is None:
            raise ManifestError(
                f"unknown extra_tags name: {key!r}. Use an integer tag code "
                f"or one of {sorted(_WELL_KNOWN_TIFF_TAGS)}."
            )
        return code, key
    raise ManifestError(f"extra_tags key must be str or int, got {key!r}")


def _apply_extra_tags_with_tifffile(
    path: pathlib.Path, extra_tags: dict[Any, Any]
) -> None:
    """Rewrite ``path`` so each entry in ``extra_tags`` lands as a real TIFF tag.

    rasterio cannot emit private numeric TIFF tags via its writer, and its
    `update_tags` API stores well-known names like ``Software`` inside the
    ``GDAL_METADATA`` XML rather than in the actual TIFF tag code. To get a
    real IFD entry, the generator writes the raster with rasterio first
    (so the GeoTIFF tags are correct) and then this helper rewrites the
    file in place via tifffile, preserving the GeoTIFF tags and adding the
    requested extras.

    Only the codes listed in ``_GEOTIFF_TAG_CODES`` survive the rewrite.
    That set covers the GeoTIFF georeferencing tags, GDAL_METADATA, and
    the SampleFormat / ExtraSamples tags rasterio writes for non-uint8
    dtypes. Other tags rasterio may emit (nodata sentinel, ResolutionUnit,
    etc.) are not currently forwarded; extend the list if a future fixture
    needs them.
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as t:
        page = t.pages[0]
        pixels = page.asarray()
        photometric = page.photometric
        planarconfig = page.planarconfig
        preserved: list[tuple[int, int, int, Any, bool]] = []
        for tag in page.tags:
            if tag.code in _GEOTIFF_TAG_CODES:
                dcode = tag.dtype.value if hasattr(tag.dtype, "value") else int(tag.dtype)
                preserved.append((tag.code, dcode, tag.count, tag.value, True))

    # tifffile reserves the `description` and `software` writer kwargs for
    # tag codes 270 and 305 respectively; routing those through `extratags`
    # produces a warning and a silently-dropped tag. Pull them out.
    description: str | None = None
    software: str | None = None
    extratags = list(preserved)
    # Resolve every key once, sort by numeric code so the on-disk order
    # is deterministic across runs.
    resolved = sorted(
        ((_normalize_extra_tag_key(k)[0], str(v)) for k, v in extra_tags.items()),
        key=lambda item: item[0],
    )
    for code, sval in resolved:
        if code == 270:
            description = sval
        elif code == 305:
            software = sval
        else:
            # Type 's' = ASCII string. count=0 lets tifffile size it.
            extratags.append((code, "s", 0, sval, True))

    write_kwargs: dict[str, Any] = {
        "photometric": photometric,
        "planarconfig": planarconfig,
        "metadata": None,  # do not emit tifffile's JSON header into tag 270
        "extratags": extratags,
    }
    if description is not None:
        write_kwargs["description"] = description
    if software is not None:
        write_kwargs["software"] = software

    tifffile.imwrite(str(path), pixels, **write_kwargs)


def _write_cog_fixture(
    entry: dict[str, Any], out_path: pathlib.Path, pixels: np.ndarray
) -> None:
    """Materialise a COG fixture by staging a plain GTiff then copying
    through GDAL's ``COG`` driver.

    The COG driver enforces tiling and IFD ordering per
    https://www.cogeo.org/spec/. Going through ``rasterio.shutil.copy``
    is the supported way to invoke it from rasterio.
    """
    import tempfile

    import rasterio
    from rasterio.shutil import copy as rio_copy

    base_kwargs = _rasterio_kwargs(entry)
    # The COG driver owns these settings; passing them through the source
    # GTiff is fine, but we strip the codec/predictor from the staging
    # write so the source is cheap and let the COG copy apply compression.
    staging_kwargs = dict(base_kwargs)
    for k in ("compress", "predictor", "zlevel", "zstd_level", "jpeg_quality"):
        staging_kwargs.pop(k, None)
    staging_kwargs["tiled"] = True
    staging_kwargs.setdefault("blockxsize", entry.get("tile_size", 16))
    staging_kwargs.setdefault("blockysize", entry.get("tile_size", 16))

    with tempfile.TemporaryDirectory() as td:
        staging = pathlib.Path(td) / f"{entry['id']}.staging.tif"
        with rasterio.open(str(staging), "w", **staging_kwargs) as dst:
            for b in range(entry["bands"]):
                dst.write(pixels[b], b + 1)
            gdal_md = entry.get("gdal_metadata") or {}
            for domain, items in sorted(gdal_md.items()):
                dst.update_tags(
                    ns=domain,
                    **{str(k): str(v) for k, v in sorted(items.items())},
                )
            extra_tags = entry.get("extra_tags") or {}
            if extra_tags:
                dst.update_tags(
                    **{str(k): str(v) for k, v in sorted(extra_tags.items())}
                )

        cog_kwargs: dict[str, Any] = {
            "driver": "COG",
            "blocksize": entry.get("tile_size", 16),
            "overview_resampling": entry.get("overview_resampling", "nearest"),
        }
        if entry["compression"] != "none":
            cog_kwargs["compress"] = entry["compression"]
            level = entry.get("compression_level")
            if isinstance(level, int) and level >= 0:
                if entry["compression"] == "deflate":
                    cog_kwargs["level"] = level
        rio_copy(str(staging), str(out_path), **cog_kwargs)


def write_fixture(entry: dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    """Materialise one fixture. Returns the written `.tif` path.

    Real writes only run when called by the non-dry-run code path.
    Fixtures with ``external_overview: true`` also emit a sidecar
    ``<id>.tif.ovr`` next to the returned path.
    """
    import rasterio

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{entry['id']}.tif"
    pixels = _make_pixels(entry)

    if entry.get("cog"):
        _write_cog_fixture(entry, out_path, pixels)
        os.utime(out_path, (DETERMINISTIC_EPOCH, DETERMINISTIC_EPOCH))
        return out_path

    kwargs = _rasterio_kwargs(entry)
    extra_tags = entry.get("extra_tags") or {}

    with rasterio.open(str(out_path), "w", **kwargs) as dst:
        for b in range(entry["bands"]):
            dst.write(pixels[b], b + 1)
        gdal_md = entry.get("gdal_metadata") or {}
        for domain, items in sorted(gdal_md.items()):
            dst.update_tags(
                ns=domain, **{str(k): str(v) for k, v in sorted(items.items())}
            )
        overviews = entry.get("overviews") or []
        if overviews and not entry.get("external_overview"):
            from rasterio.enums import Resampling
            resamp = getattr(
                Resampling, entry.get("overview_resampling", "nearest")
            )
            dst.build_overviews(sorted(overviews), resamp)

    # External overviews are built by re-opening the file in `r+` with
    # the TIFF_USE_OVR=YES env hint so GDAL writes a `<path>.ovr` sidecar
    # instead of appending an internal overview IFD. The sidecar is
    # committed alongside the .tif. The reopen mutates the .tif mtime,
    # which the final ``os.utime`` below renormalises.
    #
    # COMPRESS_OVERVIEW is hard-coded to DEFLATE because every current
    # external-overview fixture wants a compressed sidecar. If a future
    # fixture needs another codec (or none), promote this to a manifest
    # knob rather than threading another env override here.
    overviews = entry.get("overviews") or []
    if overviews and entry.get("external_overview"):
        from rasterio.enums import Resampling
        resamp = getattr(
            Resampling, entry.get("overview_resampling", "nearest")
        )
        with rasterio.Env(TIFF_USE_OVR="YES", COMPRESS_OVERVIEW="DEFLATE"):
            with rasterio.open(str(out_path), "r+") as dst:
                dst.build_overviews(sorted(overviews), resamp)
        os.utime(
            out_path.with_suffix(out_path.suffix + ".ovr"),
            (DETERMINISTIC_EPOCH, DETERMINISTIC_EPOCH),
        )

    if extra_tags:
        _apply_extra_tags_with_tifffile(out_path, extra_tags)

    # Normalise mtime so re-runs are byte-stable. The .tif is touched here
    # because both the rasterio writer above and the external-overview
    # ``r+`` reopen bump mtime to wall-clock time.
    os.utime(out_path, (DETERMINISTIC_EPOCH, DETERMINISTIC_EPOCH))
    return out_path


def generate(
    only: Iterable[str] | None = None,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    manifest_path: pathlib.Path = MANIFEST_PATH,
) -> list[pathlib.Path]:
    """Validate the manifest and (unless dry_run) write every fixture.

    Returns the list of paths that were (or would be) written, in
    sorted-id order.
    """
    manifest = load_manifest(manifest_path)
    entries = validate(manifest)

    if only is not None:
        wanted = set(only)
        entries = [e for e in entries if e["id"] in wanted]
        missing = wanted - {e["id"] for e in entries}
        if missing:
            raise ManifestError(f"unknown fixture ids: {sorted(missing)}")

    # Deterministic output order: sort by id after validation.
    entries.sort(key=lambda e: e["id"])

    paths: list[pathlib.Path] = []
    for entry in entries:
        target = output_dir / f"{entry['id']}.tif"
        if dry_run:
            paths.append(target)
            continue
        paths.append(write_fixture(entry, output_dir))
    return paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate.py",
        description="Deterministically (re)build the geotiff golden corpus.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and print planned outputs without writing.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="ID",
        help="Restrict to one fixture id; may be repeated.",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the .tif files (default: ./fixtures).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        paths = generate(
            only=args.only,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2

    label = "would write" if args.dry_run else "wrote"
    for path in paths:
        print(f"{label} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
