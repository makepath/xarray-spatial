# User Guide Notebook: Create or Refactor

Create a new xarray-spatial user guide notebook, or refactor an existing one into
the established structure. The prompt is: $ARGUMENTS

If a notebook path is given, refactor it. Otherwise create a new one.

---

## Notebook structure

Every user guide notebook follows this cell sequence:

```
 0  [markdown]  # Title (short, opinionated, ties to a real use case)
 1  [markdown]  ### What you'll build (summary + preview image + nav links)
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

1. **Run all markdown cells and code comments through `/humanizer`.**
2. Never use em dashes (`--`, `---`, or the unicode character).
3. Short and direct. Technical but not sterile.
4. Opening cell: tie the notebook to a real-world use case. Keep it grounded, not
   dramatic. Mention the topic and why it matters, skip intensity.
5. "What you'll build" cell: one sentence of what they'll do, a preview image
   (`images/filename.png`), and anchor links to each `##` section.
6. Use lists for readability when there are 3+ parallel items.
7. Section intros: 2-4 sentences max. Link to a real external reference if one
   exists. End with a short note on what the upcoming plot shows.
8. Bonus/fun sections: frame them as "just for fun" or "extra credit", separate
   from the main narrative.
9. References section at the end with real URLs, no filler.

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
6. Restructure cells to match the section pattern above.
7. Run all markdown through `/humanizer`.
8. Verify the notebook executes: `jupyter nbconvert --execute`.

---

## New notebook checklist

When creating from scratch:

1. Pick a topic and a real-world angle for the opening.
2. Write the full cell sequence following the structure above.
3. Generate a preview image and save to `images/`.
4. Run all markdown through `/humanizer`.
5. Verify the notebook executes: `jupyter nbconvert --execute`.
