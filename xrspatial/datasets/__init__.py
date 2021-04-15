import os
import xarray as xr


__all__ = ["available", "get_data"]

_module_path = os.path.dirname(os.path.abspath(__file__))
_available_datasets = [p for p in next(os.walk(_module_path))[1]
                       if not p.startswith("__")]
available_datasets = _available_datasets


def get_data(dataset):
    """
    Open example multispectral band data.
    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``xrspatial.datasets.available`` for
        all options.
    Examples
    --------
    >>>     xrspatial.datasets.get_data("sentinel-2")
    """
    data = {}
    if dataset in _available_datasets:
        folder_path = os.path.abspath(os.path.join(_module_path, dataset))
        band_files = [p for p in next(os.walk(folder_path))[2]]
        for band_file in band_files:
            array = xr.open_dataarray(os.path.join(folder_path, band_file))
            data[array.Name] = array
    else:
        msg = f'The dataset {dataset} is not available. '
        msg += f'Available folders are {available_datasets}.'
        raise ValueError(msg)
    return data
