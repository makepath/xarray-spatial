# New Issues: Feature Gap Analysis and Issue Creation

Audit the README feature matrix, identify gaps and opportunities, and file
GitHub issues for the best candidates. The prompt is: {{ARGUMENTS}}

---

## Step 1 -- Read the feature matrix

1. Read `README.md` and extract every function listed in the feature matrix tables.
2. For each function, record:
   - Category (Surface, Hydrology, Focal, etc.)
   - Backend support (which of the four columns are native, fallback, or missing)
3. Read the source files referenced in the matrix to confirm what actually exists
   (the README can drift from reality).

## Step 2 -- Identify backend gaps

1. List every function where one or more backends show 🔄 (fallback) or blank
   (unsupported).
2. Prioritize gaps where:
   - The function already has 3 of 4 backends (low effort to complete the set)
   - The missing backend is CuPy or Dask+CuPy (GPU support matters for large rasters)
   - The function is commonly used by GIS analysts (slope, aspect, flow direction, etc.)
3. Draft 1-3 maintenance issues for the highest-value backend completions.

## Step 3 -- Identify missing features

Think about what GIS analysts and Python spatial data scientists actually need
that the library does not yet provide. Consider:

- **Surface analysis gaps:** contour line extraction, profile/cross-section tools,
  terrain shadow analysis, sky-view factor, landform classification
  (Weiss 2001, Jasiewicz & Stepinski 2013)
- **Hydrology gaps:** HAND (Height Above Nearest Drainage) generation (not just
  flood-depth-from-HAND), depression filling / breach, channel width estimation,
  compound topographic index (CTI / wetness index)
- **Focal / neighborhood gaps:** directional filters, morphological operators
  (erode, dilate, open, close), texture metrics (entropy, GLCM), circular
  or annular kernels
- **Multispectral gaps:** water indices (NDWI, MNDWI), built-up indices (NDBI),
  snow index (NDSI), tasseled cap, PCA, band math DSL
- **Interpolation gaps:** natural neighbor, RBF (radial basis function),
  trend surface
- **Zonal gaps:** zonal geometry (area, perimeter, centroid), majority/minority
  filter, zonal histogram
- **Network / connectivity:** cost-path corridor, least-cost corridor,
  visibility network (intervisibility between multiple points)
- **Time series:** temporal compositing (median, max-NDVI), change detection,
  phenology metrics
- **I/O and interop:** raster clipping to polygon, raster merge/mosaic,
  coordinate reprojection helpers

Do NOT suggest features that duplicate what GDAL/rasterio already do well
unless there is a clear benefit to having a pure-Python/Numba version (e.g.
GPU support, Dask integration, no C dependency).

Select the 3-5 most impactful feature suggestions. Rank by:
1. How often GIS analysts need the operation (daily-use beats niche)
2. How well it fits the library's existing architecture
3. Whether it fills a gap no other GDAL-free Python library covers

## Step 4 -- Draft the issues

For each candidate (both maintenance and new-feature), draft a GitHub issue
following the `.github/ISSUE_TEMPLATE/feature-proposal.md` template:

- **Title:** short, imperative (e.g. "Add NDWI water index to multispectral module")
- **Labels:** `enhancement` plus any topical labels that fit
- **Body sections:**
  - Reason or Problem
  - Proposal (Design, Usage, Value)
  - Stakeholders and Impacts
  - Drawbacks
  - Alternatives
  - Unresolved Questions

Keep each issue body concise. Cite specific algorithms or papers where
relevant. Include a short code snippet showing the proposed API.

## Step 5 -- Humanize and create

1. Collect all drafted issue bodies into a batch.
2. **Run each issue body through [TOOL: humanize]** to strip AI writing
   patterns before creating the issue.
3. Create each issue with `gh issue create`, passing the humanized title,
   body, and labels.
4. Record the issue numbers and URLs.

## Step 6 -- Summary

Print a table of all created issues:

```
| # | Title | Labels | URL |
|---|-------|--------|-----|
```

Then briefly explain the rationale: why these issues were chosen, what
analyst workflows they unblock, and any issues you considered but dropped
(with a one-line reason for each).

---

## General rules

- Do not create duplicate issues. Before filing, search existing issues with
  `gh issue list --limit 100 --state all` and skip anything already covered.
- Run [TOOL: humanize] on every issue title and body before creating it.
- If {{ARGUMENTS}} contains specific focus areas (e.g. "hydrology only"),
  restrict the analysis to those categories.
- If {{ARGUMENTS}} is empty, run the full analysis across all categories.
- Prefer fewer, higher-quality issues over a long wishlist.
