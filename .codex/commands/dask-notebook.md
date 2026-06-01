# Dask ETL Notebook

Create a Jupyter notebook that sets up a Dask distributed LocalCluster and walks
through an ETL (Extract, Transform, Load) workflow. The prompt is: $ARGUMENTS

Use the prompt to determine the data domain, transformations, and output format.
If no prompt is given, use a geospatial raster ETL as the default domain
(consistent with the xarray-spatial project).

---

## Notebook structure

Every Dask ETL notebook follows this cell sequence:

```
 0  [markdown]  # Title + one-line description of the pipeline
 1  [markdown]  ### Overview (what the pipeline does, what you'll learn)
 2  [markdown]  One-liner about the imports
 3  [code    ]  Imports
 4  [markdown]  ## Cluster Setup
 5  [code    ]  Create and inspect a dask.distributed LocalCluster + Client
 6  [markdown]  Brief note on the dashboard URL and how to read it
 7  [markdown]  ## Extract
 8  [code    ]  Load or generate source data as lazy Dask arrays
 9  [markdown]  Describe the raw data: shape, dtype, chunk layout
10  [code    ]  Inspect / visualize a sample of the raw data
11  [markdown]  ## Transform
12  [code    ]  Apply transformations (filtering, rechunking, computation)
13  [markdown]  Explain what the transform does and why it benefits from Dask
14  [code    ]  (Optional) Additional transform step(s)
15  [markdown]  ## Load
16  [code    ]  Write results to disk (Zarr, Parquet, GeoTIFF, etc.)
17  [markdown]  Confirm output and show summary statistics
18  [code    ]  Read back and verify the output
19  [markdown]  ## Cleanup
20  [code    ]  Close the client and cluster
21  [markdown]  ### Summary + next steps
```

Sections can be repeated or extended when the prompt calls for more transform
steps. The core requirement is that every notebook has all five phases: Cluster
Setup, Extract, Transform, Load, Cleanup.

---

## Cluster Setup cell

Always use this pattern for the cluster:

```python
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="2GB",
)
client = Client(cluster)
client
```

Include a markdown cell after the cluster cell noting:
- The dashboard link (usually `http://localhost:8787/status`)
- That `n_workers` and `memory_limit` should be tuned for the machine

If the prompt asks for a specific cluster configuration (GPU workers, adaptive
scaling, remote scheduler), adjust accordingly but keep the default simple.

---

## Code conventions

### Imports

Standard import block for a Dask ETL notebook:

```python
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
```

Add extras only when needed (e.g. `import pandas as pd`, `import rioxarray`,
`from xrspatial import slope`). Keep the import cell minimal.

### Dask best practices to demonstrate

- **Lazy by default**: build the computation graph before calling `.compute()`.
  Show the repr of a lazy array at least once so the reader sees the task graph.
- **Chunking**: explain chunk choices. Use `dask.array.from_array(..., chunks=)`
  or `xr.open_dataset(..., chunks={})` depending on the source.
- **Avoid full materialization mid-pipeline**: no `.values` or `.compute()` until
  the Load phase unless there is a good reason (and if so, explain why).
- **Persist when reused**: if an intermediate result is used in multiple
  downstream steps, call `client.persist(result)` and explain why.
- **Progress feedback**: use `dask.diagnostics.ProgressBar` or point the reader
  to the dashboard.

### Data handling

- Generate or load data lazily. For synthetic data, use `dask.array.random` or
  wrap numpy arrays with `da.from_array(..., chunks=...)`.
- For file-based sources, prefer `xr.open_dataset` / `xr.open_mfdataset` with
  explicit `chunks=` to get lazy Dask-backed arrays.
- For the Load phase, prefer Zarr (`to_zarr()`) as the default output format
  since it supports parallel writes natively. Mention Parquet or GeoTIFF as
  alternatives when relevant.

### Cleanup

Always close the client and cluster at the end:

```python
client.close()
cluster.close()
```

---

## Writing rules

1. **Run all markdown cells and code comments through `/humanizer`.**
2. Never use em dashes.
3. Short and direct. Technical but not sterile.
4. Title cell (h1): describe the pipeline, e.g.
   `Dask ETL: Raster Slope Analysis at Scale` or
   `Dask ETL: Aggregating Sensor Readings to Parquet`.
5. Overview cell: 2-3 sentences on what the pipeline does and what Dask concepts
   the reader will pick up. No hype.
6. Each phase (Extract, Transform, Load) gets a brief markdown intro (2-4
   sentences) explaining what happens and why.
7. Use inline comments in code cells sparingly. Let the markdown cells carry the
   explanation.

---

## Checklist

When creating the notebook:

1. Pick a data domain from the prompt (or default to geospatial raster).
2. Write the full cell sequence following the structure above.
3. Verify all code cells are syntactically correct and self-contained.
4. Run all markdown through `/humanizer`.
5. Ensure the notebook cleans up after itself (cluster closed, temp files noted).
