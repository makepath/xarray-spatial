"""Check a raster against the xarray-spatial input contract.

xrspatial operations share an implicit contract about their input
DataArray: it must be an ``xr.DataArray`` whose last two dims are the
spatial axes (2-D, or 3-D with a leading band/time axis), with a real
numeric dtype, numeric/monotonic/evenly-spaced coordinates, a positive
finite cell size, lat/lon in geographic range, and a CRS for geodesic
methods and GeoTIFF I/O. Those rules live in the per-op validators in
``utils.py`` and surface one at a time when an op raises mid-pipeline.

``validate`` checks all of them up front and returns a report of every
violation with a suggested fix, rather than raising on the first one.
Every check inspects structure only (dims, coords, dtype, attrs) and
never reads the data buffer, so dask and cupy arrays validate without
materializing and all four backends behave identically.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = [
    "validate",
    "validate_dataset",
    "ValidationIssue",
    "ValidationReport",
    "DatasetValidationReport",
    "XrsContractError",
]

# Names that mark a dimension as a recognised spatial axis. Mirrors the
# lat/lon name sets used by the geodesic helpers in utils.py.
_Y_DIM_NAMES = {"y", "lat", "latitude"}
_X_DIM_NAMES = {"x", "lon", "longitude"}
_LAT_NAMES = {"lat", "latitude"}
_LON_NAMES = {"lon", "longitude"}


class XrsContractError(ValueError):
    """Raised by ``validate(raise_on_error=True)`` when errors are found."""


@dataclass(frozen=True)
class ValidationIssue:
    """One way an array is out of compliance with the contract.

    Parameters
    ----------
    severity : str
        ``'error'`` (a spatial op will fail) or ``'warning'`` (behavior
        degrades, e.g. distance-based results may be inaccurate).
    check : str
        Short identifier for the failed check, e.g. ``'dtype'``.
    message : str
        What is wrong.
    suggestion : str
        How to get back into compliance.
    """

    severity: str
    check: str
    message: str
    suggestion: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message} Fix: {self.suggestion}"


class ValidationReport:
    """Result of validating a single DataArray against the contract.

    Truthy when the array has no error-level issues (warnings are
    allowed), so ``if not da.xrs.validate(): ...`` reads naturally.
    """

    def __init__(self, issues, name=None):
        self.issues = list(issues)
        self.name = name

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """True when there are no error-level issues."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.is_valid

    def raise_if_errors(self):
        """Raise :class:`XrsContractError` listing every error-level issue."""
        if not self.errors:
            return
        label = f" for {self.name!r}" if self.name else ""
        lines = [f"raster{label} is not contract-compliant:"]
        for issue in self.errors:
            lines.append(f"  - {issue.message} Fix: {issue.suggestion}")
        raise XrsContractError("\n".join(lines))

    def __repr__(self) -> str:
        label = f" for {self.name!r}" if self.name else ""
        if not self.issues:
            return f"ValidationReport{label}: compliant (0 issues)"
        n_err = len(self.errors)
        n_warn = len(self.warnings)
        header = (
            f"ValidationReport{label}: "
            f"{n_err} error(s), {n_warn} warning(s)"
        )
        lines = [header]
        for issue in self.issues:
            lines.append(
                f"  [{issue.severity}] {issue.message} "
                f"Fix: {issue.suggestion}"
            )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        label = (
            f" for <code>{html.escape(str(self.name))}</code>"
            if self.name
            else ""
        )
        if not self.issues:
            return (
                f"<div><b>ValidationReport</b>{label}: "
                "compliant (0 issues)</div>"
            )
        rows = []
        for issue in self.issues:
            rows.append(
                "<tr>"
                f"<td><b>{html.escape(issue.severity)}</b></td>"
                f"<td>{html.escape(issue.check)}</td>"
                f"<td>{html.escape(issue.message)}</td>"
                f"<td>{html.escape(issue.suggestion)}</td>"
                "</tr>"
            )
        n_err = len(self.errors)
        n_warn = len(self.warnings)
        return (
            f"<div><b>ValidationReport</b>{label}: "
            f"{n_err} error(s), {n_warn} warning(s)"
            "<table><tr><th>severity</th><th>check</th>"
            "<th>issue</th><th>suggestion</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )


class DatasetValidationReport:
    """Per-variable validation result for a Dataset.

    Holds an ordered ``{var_name: ValidationReport}`` map. Truthy when
    every variable is compliant.
    """

    def __init__(self, reports):
        # reports: dict mapping variable name -> ValidationReport
        self.reports = dict(reports)

    @property
    def is_valid(self) -> bool:
        return all(r.is_valid for r in self.reports.values())

    def __bool__(self) -> bool:
        return self.is_valid

    def raise_if_errors(self):
        """Raise if any variable has error-level issues."""
        offending = {
            name: r for name, r in self.reports.items() if not r.is_valid
        }
        if not offending:
            return
        lines = ["Dataset is not contract-compliant:"]
        for name, r in offending.items():
            for issue in r.errors:
                lines.append(
                    f"  - [{name}] {issue.message} Fix: {issue.suggestion}"
                )
        raise XrsContractError("\n".join(lines))

    def __repr__(self) -> str:
        if not self.reports:
            return "DatasetValidationReport: no data variables"
        status = "compliant" if self.is_valid else "non-compliant"
        lines = [f"DatasetValidationReport: {status}"]
        for name, r in self.reports.items():
            if r.is_valid and not r.warnings:
                lines.append(f"  {name}: compliant")
            else:
                lines.append(f"  {name}:")
                for issue in r.issues:
                    lines.append(
                        f"    [{issue.severity}] {issue.message} "
                        f"Fix: {issue.suggestion}"
                    )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        status = "compliant" if self.is_valid else "non-compliant"
        parts = [f"<div><b>DatasetValidationReport</b>: {status}"]
        for name, r in self.reports.items():
            parts.append(f"<div><b>{html.escape(str(name))}</b></div>")
            parts.append(r._repr_html_())
        parts.append("</div>")
        return "".join(parts)


def _is_real_numeric(dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype, np.complexfloating
    )


def _discover_crs(agg):
    """Return a CRS for *agg*, or None. Mirrors the polygonize convention.

    Resolution order: ``attrs['crs']``, ``attrs['crs_wkt']``, then
    ``agg.rio.crs`` if rioxarray is installed. The xrspatial.geotiff
    "no georeference" marker forces None.
    """
    if agg.attrs.get("_xrspatial_no_georef"):
        return None
    crs = agg.attrs.get("crs")
    if crs is not None:
        return crs
    crs_wkt = agg.attrs.get("crs_wkt")
    if crs_wkt is not None:
        return crs_wkt
    try:
        rio_crs = agg.rio.crs
    except Exception:
        return None
    return rio_crs


def _coord_values(coord):
    """1-D coordinate values as a host numpy array, or None.

    Coordinates are metadata, not the data buffer, so reading them does
    not materialize a dask/cupy raster. A cupy-backed coordinate is
    pulled to host with ``.get()``.
    """
    if coord is None or coord.ndim != 1:
        return None
    data = coord.data
    get = getattr(data, "get", None)
    if get is not None and not isinstance(data, np.ndarray):
        # cupy ndarray -> host
        try:
            data = get()
        except TypeError:
            data = np.asarray(coord.values)
    values = np.asarray(data)
    if not np.issubdtype(values.dtype, np.number):
        return None
    return values


def validate(agg) -> ValidationReport:
    """Check a DataArray against the xarray-spatial input contract.

    Parameters
    ----------
    agg : xarray.DataArray
        Raster to check.

    Returns
    -------
    ValidationReport
        Lists every contract violation as an error (a spatial op will
        fail) or a warning (behavior degrades), each with a suggested
        fix. The report is truthy when there are no error-level issues.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np, xarray as xr
        >>> import xrspatial  # registers the .xrs accessor
        >>> bad = xr.DataArray(
        ...     np.array([["a", "b"], ["c", "d"]]), dims=["y", "x"]
        ... )
        >>> report = bad.xrs.validate()
        >>> bool(report)
        False
    """
    issues = []

    if not isinstance(agg, xr.DataArray):
        issues.append(
            ValidationIssue(
                "error",
                "type",
                f"input is a {type(agg).__name__}, not an xarray.DataArray.",
                "wrap the array with xr.DataArray(...).",
            )
        )
        return ValidationReport(issues, name=None)

    name = agg.name

    # --- ndim ---
    if agg.ndim not in (2, 3):
        issues.append(
            ValidationIssue(
                "error",
                "ndim",
                f"array is {agg.ndim}D; spatial ops need 2D, or 3D with a "
                "leading band/time axis.",
                "reduce to 2D/3D, e.g. select a band/time slice with "
                "da.isel(...).",
            )
        )

    # --- dtype ---
    if not _is_real_numeric(agg.dtype):
        issues.append(
            ValidationIssue(
                "error",
                "dtype",
                f"dtype is {agg.dtype}; spatial ops need a real numeric "
                "(integer or float) dtype.",
                "cast with da.astype('float64').",
            )
        )

    # Spatial-axis checks only make sense once there are >= 2 dims.
    if agg.ndim >= 2:
        ydim, xdim = agg.dims[-2], agg.dims[-1]

        # --- spatial dim names ---
        if (
            str(ydim).lower() not in _Y_DIM_NAMES
            or str(xdim).lower() not in _X_DIM_NAMES
        ):
            issues.append(
                ValidationIssue(
                    "warning",
                    "spatial_dims",
                    f"last two dims are ('{ydim}', '{xdim}'); xrspatial "
                    "expects spatial axes named like ('y', 'x') or "
                    "('lat', 'lon').",
                    "rename with da.rename({'%s': 'y', '%s': 'x'})."
                    % (ydim, xdim),
                )
            )

        _check_spatial_coords(agg, ydim, xdim, issues)

        # --- crs ---
        if _discover_crs(agg) is None:
            issues.append(
                ValidationIssue(
                    "warning",
                    "crs",
                    "no CRS found; geodesic methods and GeoTIFF I/O cannot "
                    "georeference the raster.",
                    "set one, e.g. da.attrs['crs'] = 'EPSG:4326'.",
                )
            )

    return ValidationReport(issues, name=name)


def _check_spatial_coords(agg, ydim, xdim, issues):
    """Coordinate-dependent checks: presence, monotonic, spacing, cellsize."""
    have_both = True
    for dim in (ydim, xdim):
        # A bare dimension is absent from ``coords`` even though
        # ``coords.get(dim)`` synthesizes a default range index, so test
        # membership explicitly.
        if dim not in agg.coords:
            have_both = False
            issues.append(
                ValidationIssue(
                    "warning",
                    "coords_present",
                    f"dim '{dim}' has no coordinate; resolution-dependent "
                    "ops (slope, hillshade, proximity) cannot infer cell "
                    "size.",
                    f"assign a coordinate, e.g. "
                    f"da.assign_coords({dim}=np.arange(da.sizes['{dim}'])).",
                )
            )
            continue
        coord = agg.coords[dim]
        if not np.issubdtype(coord.dtype, np.number):
            have_both = False
            issues.append(
                ValidationIssue(
                    "error",
                    "coords_numeric",
                    f"'{dim}' coordinate is {coord.dtype}, not numeric.",
                    f"cast it, e.g. da['{dim}'] = "
                    f"da['{dim}'].astype('float64').",
                )
            )
            continue

        values = _coord_values(coord)
        if values is None or values.size < 2:
            continue

        diffs = np.diff(values)
        if not np.all(np.isfinite(diffs)):
            continue

        # --- monotonic ---
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            have_both = False
            issues.append(
                ValidationIssue(
                    "error",
                    "monotonic",
                    f"'{dim}' coordinate is not strictly monotonic.",
                    f"sort with da.sortby('{dim}').",
                )
            )
            continue

        # --- even spacing --- (mirrors utils._warn_if_irregular_spacing,
        # which compares each step to the averaged span/(n-1) resolution)
        if values.size >= 3:
            step = abs(values[-1] - values[0]) / (values.size - 1)
            if not np.allclose(np.abs(diffs), step, rtol=1e-5, atol=0):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "even_spacing",
                        f"'{dim}' coordinate is not evenly spaced; "
                        "distance-based results may be inaccurate.",
                        "resample to a regular grid, or set an explicit "
                        "da.attrs['res'] = (dx, dy).",
                    )
                )

    # --- cellsize --- needs both spatial coords usable.
    if have_both:
        _check_cellsize(agg, ydim, xdim, issues)

    _check_geographic_range(agg, ydim, xdim, issues)


def _check_cellsize(agg, ydim, xdim, issues):
    """Cell size must be positive and finite on both axes."""
    yv = _coord_values(agg.coords.get(ydim))
    xv = _coord_values(agg.coords.get(xdim))
    if yv is None or xv is None or yv.size < 2 or xv.size < 2:
        return
    cellsize_y = (yv[-1] - yv[0]) / (yv.size - 1)
    cellsize_x = (xv[-1] - xv[0]) / (xv.size - 1)
    for axis, cellsize in (("x", cellsize_x), ("y", cellsize_y)):
        if not np.isfinite(cellsize) or cellsize == 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "cellsize",
                    f"{axis}-axis cell size is {cellsize}; planar ops need "
                    "a positive, finite cell size.",
                    "fix the coordinate spacing, or set "
                    "da.attrs['res'] = (dx, dy).",
                )
            )


def _check_geographic_range(agg, ydim, xdim, issues):
    """When dims look like lat/lon, values must be in geographic range.

    Only real coordinates are checked: a bare dimension synthesizes a
    default integer index whose range would otherwise trip a false
    "looks projected" warning.
    """
    is_lat = str(ydim).lower() in _LAT_NAMES and ydim in agg.coords
    is_lon = str(xdim).lower() in _LON_NAMES and xdim in agg.coords
    if not (is_lat or is_lon):
        return
    if is_lat:
        lat = _coord_values(agg.coords[ydim])
        if lat is not None and lat.size:
            lo, hi = float(np.nanmin(lat)), float(np.nanmax(lat))
            if lo < -90 or hi > 90 or (hi - lo) > 180:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "geographic_range",
                        f"'{ydim}' values [{lo}, {hi}] fall outside "
                        "latitude range [-90, 90]; coordinates look "
                        "projected, not geographic.",
                        "use geographic (lat/lon) coordinates for geodesic "
                        "methods, or rename the projected axis.",
                    )
                )
    if is_lon:
        lon = _coord_values(agg.coords[xdim])
        if lon is not None and lon.size:
            lo, hi = float(np.nanmin(lon)), float(np.nanmax(lon))
            if lo < -180 or hi > 360 or (hi - lo) > 360:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "geographic_range",
                        f"'{xdim}' values [{lo}, {hi}] fall outside "
                        "longitude range [-180, 360]; coordinates look "
                        "projected, not geographic.",
                        "use geographic (lat/lon) coordinates for geodesic "
                        "methods, or rename the projected axis.",
                    )
                )


def validate_dataset(ds) -> DatasetValidationReport:
    """Check each data variable of a Dataset against the contract.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose data variables are rasters.

    Returns
    -------
    DatasetValidationReport
        A per-variable map of :class:`ValidationReport`. Truthy when
        every variable is compliant.
    """
    reports = {name: validate(ds[name]) for name in ds.data_vars}
    return DatasetValidationReport(reports)
