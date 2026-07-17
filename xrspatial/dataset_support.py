"""Decorators for transparent xr.Dataset support on xr.DataArray functions."""

from __future__ import annotations

import functools
import inspect

import xarray as xr


def supports_dataset(func):
    """Decorator that lets single-input DataArray functions accept a Dataset.

    When a Dataset is passed as the first argument, the wrapped function
    is called on each data variable and the results are collected into
    a new Dataset.
    """
    sig = inspect.signature(func)
    has_name_param = 'name' in sig.parameters
    first_param = next(iter(sig.parameters))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the raster whether it was passed positionally or by its
        # real keyword name (e.g. cost_distance(raster=...)); a wrapper
        # that only accepts it positionally breaks keyword callers.
        if args:
            agg, rest = args[0], args[1:]
        elif first_param in kwargs:
            agg, rest = kwargs.pop(first_param), ()
        else:
            # Missing entirely: let func raise its own TypeError
            return func(*args, **kwargs)
        if isinstance(agg, xr.Dataset):
            results = {}
            for var_name in agg.data_vars:
                kw = dict(kwargs)
                if has_name_param:
                    kw['name'] = var_name
                results[var_name] = func(agg[var_name], *rest, **kw)
            return xr.Dataset(results, attrs=agg.attrs)
        return func(agg, *rest, **kwargs)

    return wrapper


def supports_dataset_bands(**band_param_map):
    """Decorator for multi-input functions that take separate band DataArrays.

    Enables passing a single Dataset with keyword arguments that map
    band aliases to Dataset variable names.

    Example::

        @supports_dataset_bands(nir='nir_agg', red='red_agg')
        def ndvi(nir_agg, red_agg, name='ndvi'): ...

        # Enables:
        ndvi(ds, nir='band_8', red='band_4')
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], xr.Dataset):
                ds = args[0]
                func_kwargs = {}
                used = set()
                for alias, param in band_param_map.items():
                    if alias not in kwargs:
                        raise TypeError(
                            f"'{alias}' keyword required when passing a Dataset"
                        )
                    var_name = kwargs[alias]
                    if var_name not in ds.data_vars:
                        raise ValueError(
                            f"'{var_name}' not in Dataset. "
                            f"Available: {list(ds.data_vars)}"
                        )
                    func_kwargs[param] = ds[var_name]
                    used.add(alias)
                # Pass through remaining kwargs (name, soil_factor, etc.)
                for k, v in kwargs.items():
                    if k not in used:
                        func_kwargs[k] = v
                return func(**func_kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
