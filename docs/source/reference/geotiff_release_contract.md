(reference.geotiff_release_contract)=

# GeoTIFF release contract

This page is the user-facing release contract for the GeoTIFF module. It defines
what each support tier promises and lists every feature exposed through
`xrspatial.geotiff.SUPPORTED_FEATURES` against its tier.

The tier strings here match the strings in `xrspatial.geotiff.SUPPORTED_FEATURES`
at runtime. If a feature is not in the table below, it is unsupported for this
release.

See also: the {ref}`release gate / audit checklist <reference.geotiff_release_gate>`
for the per-feature regression test that locks each promise, and the main
{ref}`GeoTIFF reference <reference.geotiff>` for the API surface.

## What each tier promises

### stable

A `stable` feature is part of the release contract. It is covered by regression
tests in CI, and a behavioural regression fails the build. The on-disk format,
the metadata round-trip, and the public function signature are expected to keep
working across patch releases. Removals or breaking changes go through a
deprecation window. You can rely on these in production.

### advanced

An `advanced` feature works and is tested, but its surface may change before a
1.0 release. Behaviour is documented and exercised by tests, but the contract is
narrower than `stable`. Edge cases on rare codecs, transport quirks on HTTP, and
metadata interop with the wider GDAL ecosystem are not all pinned down. Use
these in production with the expectation that you may need to follow a
release-note migration.

### experimental

An `experimental` feature exists and is covered by smoke tests, but it carries
no behavioural promise. Output may change without a deprecation window, error
paths may shift, and cross-backend numerical parity is not claimed. Some
experimental features require an explicit opt-in flag
(`allow_experimental_codecs=True` for the experimental codecs, for example) so
they do not get used by accident. Treat experimental features as something to
evaluate, not depend on.

### internal-only

An `internal_only` feature exists for one specific xrspatial use case. It is not
interoperable with the wider GeoTIFF ecosystem (GDAL, libtiff, rasterio) and is
not part of the public surface. The JPEG-in-TIFF writer is the current example:
it requires the dedicated `allow_internal_only_jpeg=True` opt-in and is
documented as not externally readable. Do not build on internal-only features.

### unsupported

`unsupported` is a doc-only tier. It captures combinations that are out of
contract for this release. Calling them either fails with an actionable error
or is documented as not supported. The list is here so release notes do not
accidentally imply coverage that does not exist. `unsupported` is not a key in
`xrspatial.geotiff.SUPPORTED_FEATURES`; it lives only on this page.

## Feature tier table

The table groups every entry in `xrspatial.geotiff.SUPPORTED_FEATURES` by
category. The `Key` column matches the runtime key.

### Codecs

| Key | Tier | Notes |
| --- | --- | --- |
| `codec.none` | stable | Uncompressed; lossless byte-for-byte round-trip. |
| `codec.deflate` | stable | Lossless; level 1-9. |
| `codec.lzw` | stable | Lossless. |
| `codec.packbits` | stable | Lossless. |
| `codec.zstd` | stable | Lossless; level 1-22. |
| `codec.lerc` | experimental | Requires `allow_experimental_codecs=True`. |
| `codec.jpeg2000` | experimental | Requires `allow_experimental_codecs=True`. |
| `codec.j2k` | experimental | Alias for `jpeg2000`; same opt-in. |
| `codec.lz4` | experimental | Requires `allow_experimental_codecs=True`; level 0-16. |
| `codec.jpeg` | internal-only | Requires `allow_internal_only_jpeg=True`. Not externally readable as GeoTIFF. |

### Readers

| Key | Tier | Notes |
| --- | --- | --- |
| `reader.local_file` | stable | Local axis-aligned GeoTIFF read; pixel bytes, transform, CRS, and nodata round-trip. |
| `reader.local_cog` | stable | Local COG overview-IFD parsing; CPU path. |
| `reader.fsspec` | advanced | fsspec-backed reads (s3, gcs, and so on). |
| `reader.http` | advanced | Plain HTTP/HTTPS reads; SSRF and private-host filters apply. |
| `reader.http_cog` | advanced | HTTP COG with range-request fetching. The transport surface (redirects, retries) is not yet contracted at the stable bar. |
| `reader.vrt` | advanced | Simple VRT mosaics. Full GDAL VRT parity is out of scope. |
| `reader.sidecar_ovr` | advanced | External `.tif.ovr` sidecar overviews. |
| `reader.allow_rotated` | advanced | Opt-in `allow_rotated=True`; drops the axis-aligned `transform` attr in favour of `rotated_affine`. |
| `reader.allow_unparseable_crs` | advanced | Opt-in escape hatch for CRS strings pyproj cannot parse. |
| `reader.gpu` | experimental | GPU read path; no cross-backend numerical parity claim. |

### Writers

| Key | Tier | Notes |
| --- | --- | --- |
| `writer.local_file` | stable | Local GeoTIFF write; round-trips bit-exact under every stable codec. |
| `writer.cog` | stable | CPU writer emits a spec-conforming COG layout (IFD-first, tiled, internal overviews, lossless codec). |
| `writer.overviews` | advanced | Internal overview IFD generation. |
| `writer.bigtiff` | advanced | `bigtiff=True` (or auto-promotion above 4 GiB) writes BigTIFF magic, 8-byte offsets, and 20-byte IFD entries. |
| `writer.bigtiff_cog` | advanced | BigTIFF plus COG. Tracked separately because the combination has its own external-interop surface. |
| `writer.gpu` | experimental | GPU write path. |
| `writer.gdal_metadata_xml` | experimental | `attrs['gdal_metadata_xml']` is escaped before serialisation. |
| `writer.extra_tags` | experimental | Pass-through of TIFF tags outside the structured set via `attrs['extra_tags']`. |

### Unsupported for this release

The following combinations are out of contract. Each bullet is tagged
**(fails loudly)** when the call raises an actionable error, or
**(documented-only)** when the call returns without an error but does not
do what a GDAL user might expect. These items will not move into the
table above without a separate epic.

- **(documented-only)** Full GDAL VRT parity. `reader.vrt` covers simple
  mosaics only; unsupported VRT XML elements are ignored rather than
  rejected.
- **(documented-only)** Warped or reprojection VRTs. Reprojection lives in
  `xrspatial.reproject`; the VRT reader does not interpret warp tags.
- **(fails loudly)** Rotated or sheared GeoTIFF *write*. Reads can opt into
  rotated transforms (`reader.allow_rotated`), but `to_geotiff` does not
  yet emit `ModelTransformationTag`.
- **(fails loudly)** Silent flattening of mixed metadata. Conflicting
  attrs across stacked inputs are surfaced rather than collapsed without
  warning.
- **(fails loudly)** File-like destinations with `cog=True`. The COG
  writer requires an on-disk path so the IFD-first layout invariant is
  checkable; both the eager and GPU writers raise
  `"cog=True is not supported for file-like destinations"` up front.

## Keeping this page honest

The tier strings on this page must match `xrspatial.geotiff.SUPPORTED_FEATURES`
at runtime. When a feature is promoted or demoted in `_attrs.py`, this page is
updated in the same PR. The release-gate page
({ref}`reference.geotiff_release_gate`) holds the per-feature regression test
that locks each `stable` promise.

Parent epic:
[#2340](https://github.com/xarray-contrib/xarray-spatial/issues/2340).
