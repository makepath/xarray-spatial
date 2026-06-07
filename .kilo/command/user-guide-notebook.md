# User Guide Notebook: Create or Refactor

Create a new xarray-spatial user guide notebook, or refactor an existing one into
the established structure. The prompt is: {{ARGUMENTS}}

If a notebook path is given, refactor it. Otherwise create a new one.

---

## Notebook structure

Every user guide notebook follows this cell sequence:

```
 0  [markdown]  # Title + subtitle (see title format below)
 1  [markdown]  ### What you'll build (summary + eye-candy preview image + nav links)
 2  [markdown]  One-liner about the imports
 3  [code    ]  Imports
 4  [markdown]  ## Data section header
 5  [code    ]  Generate or load data (ONE call, reused everywhere)
 6  [markdown]  Brief description of the raw data
 7  [code    ]  Show the data with a different colormap
      ...        Individual analysis sections (repeat pattern below)
      ...        Composite / combined section if multiple factors
      ...        Bonus visualization section (optional, for fun)
 N  [markdown]  ### References (with real URLs)
```

### Individual analysis section pattern

Each analysis gets exactly this:

1. **Markdown intro**: `## Section name`, 2-4 sentences of context with a link to
   a real reference if one exists, then a note on what the plot shows.
2. **Code cell**: compute the result, plot it overlaid on hillshade (or base layer),
   include a legend.
3. **Markdown result description** (optional, 1-2 sentences): only if the output
   needs explanation.
4. **Alert box** (optional): a GIS caveat relevant to the tool just shown, if
   there is one worth flagging that the section didn't already cover.

---

## Code conventions

### Plotting

- Use `xr.DataArray.plot.imshow()` for everything. No raw `ax.imshow(data.values)`.
- Overlay pattern:
  ```python
  fig, ax = plt.subplots(figsize=(10, 7.5))
  base.plot.imshow(ax=ax, cmap='gray', add_colorbar=False)
  overlay.plot.imshow(ax=ax, cmap=cmap, alpha=200/255, add_colorbar=False)
  ax.set_axis_off()
  ```
- Every overlay plot gets a legend via `matplotlib.patches.Patch`:
  ```python
  from matplotlib.patches import Patch
  ax.legend(handles=[Patch(facecolor='red', alpha=0.78, label='Label')],
            loc='lower right', fontsize=11, framealpha=0.9)
  ```
- Use `add_colorbar=True` with `cbar_kwargs` only for quantitative maps (risk
  scores, continuous values). Use `add_colorbar=False` for categorical overlays.
- Standard figure size: `figsize=(10, 7.5)`. Standalone plots: `size=7.5, aspect=W/H`.

### Colormaps and colorblind safety

- Never pair red and green. Use orange/blue, orange/purple, or red/blue instead.
- For risk/heat maps: `inferno` (perceptually uniform, all CVD types).
- For single-color categorical overlays: `ListedColormap(['color'])`.
- RGB images: `dims=['y', 'x', 'band']` with float values in [0, 1].

### Data handling

- Generate or load data exactly once. Reuse the same array for all sections.
- Use `xarray.where()` for filtering/masking, not manual numpy boolean indexing.
- Handle NaN edges: `fillna(0)` before integer casting, explicit NaN masks for
  RGB arrays.
- For hillshade: xrspatial returns values in [0, 1], not [0, 255].

### Imports

Standard import block:
```python
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import xrspatial
```

Add extras (e.g. `hsv_to_rgb`) only when needed.

---

## Writing rules

1. **Run all markdown cells and code comments through [TOOL: humanize].**
2. Never use em dashes (`--`, `---`, or the unicode character).
3. Short and direct. Technical but not sterile.
4. Opening cell has a title and subtitle:
   - **Title** (h1): `Xarray-Spatial {parent module}: {list a few tools covered}`.
     Examples: `Xarray-Spatial Surface: Slope, aspect, and curvature`,
     `Xarray-Spatial Proximity: Distance, allocation, and direction`,
     `Xarray-Spatial Focal: Mean, TPI, focal stats, and hotspots`.
   - **Subtitle** (plain text below the title): 2-3 sentences tying the tools to a
     real-world use case. Keep it grounded, not dramatic. Mention the topic and why
     it matters, skip intensity.
5. "What you'll build" cell: an ordered list summarizing the steps/sections the
   reader will work through, an eye-candy preview image (`images/filename.png`),
   and anchor links to each `##` section. The preview should be the most visually
   striking output from the notebook. Generate it by running the relevant code
   with `matplotlib.use('Agg')` and
   `fig.savefig('examples/user_guide/images/name.png', bbox_inches='tight', dpi=120)`.
6. Use lists for readability when there are 3+ parallel items.
7. Section intros: 2-4 sentences max. Link to a real external reference if one
   exists. End with a short note on what the upcoming plot shows.
8. Bonus/fun sections: frame them as "just for fun" or "extra credit", separate
   from the main narrative.
9. References section at the end with real URLs, no filler.

---

## GIS alert boxes

After writing each section, evaluate whether it needs a GIS caveat the reader
should know *now that they've seen the tool in action*. If so, add an alert box
as the last cell of that section (after the code output and any result
description). Not every section needs one. Skip the alert if the section's
prose or code already covers the point. The goal is to catch gotchas the reader
might hit when applying the tool to their own data, not to repeat what was just
demonstrated.

Use Jupyter's built-in alert styling:

```html
<div class="alert alert-block alert-warning">
<b>Short label.</b> Concise explanation of the caveat. Keep it practical,
not a legal disclaimer.
</div>
```

Alert types:
- `alert-warning` (yellow): caveats, gotchas, assumptions that can bite you
- `alert-info` (blue): tips, suggestions, "you might also want to look at X"
- `alert-danger` (red): things that will silently give wrong results

Common GIS topics worth flagging (only when relevant and not already covered):

- **Map projection**: Euclidean tools on lat/lon coords give results in degrees.
  Mention `GREAT_CIRCLE` or recommend reprojecting to meters.
- **2D vs 3D distance**: raster proximity ignores terrain relief.
  Point to `xrspatial.surface_distance` for terrain-following distance.
- **Resolution and units**: cell size affects results. Slope depends on the
  ratio of elevation units to cell-spacing units.
- **Edge effects**: convolution-based tools lose data at raster edges.
  Mention `boundary="nearest"` or similar padding.
- **Coordinate order**: xrspatial expects `dims=['y', 'x']` with y as rows.
  Transposed data silently produces wrong results.

Write the alert text in the same direct, non-AI style as the rest of the
notebook. Run it through [TOOL: humanize] like everything else.

---

## File organization

- Preview images go in `examples/user_guide/images/`.
- One notebook per topic. If a notebook covers too many things, split it.
- Notebooks are self-contained: own imports, own data generation.

---

## Refactoring checklist

When refactoring an existing notebook:

1. Read the entire notebook first.
2. Replace any `ax.imshow(data.values, ...)` with `data.plot.imshow(ax=ax, ...)`.
3. Consolidate data generation to a single call.
4. Add legends to all overlay plots.
5. Fix any red/green color pairings.
6. Add GIS alert boxes for relevant caveats (projection, units, edge effects).
7. Restructure cells to match the section pattern above.
8. Run all markdown through [TOOL: humanize].
9. Verify the notebook executes: `jupyter nbconvert --execute`.

---

## New notebook checklist

When creating from scratch:

1. Pick a topic and a real-world angle for the opening.
2. Write the full cell sequence following the structure above.
3. Generate a preview image and save to `images/`.
4. Add GIS alert boxes for relevant caveats (projection, units, edge effects).
5. Run all markdown through [TOOL: humanize].
6. Verify the notebook executes: `jupyter nbconvert --execute`.
