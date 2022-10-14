import tdt
import xarray as xr
from scipy.stats import mode
from scipy.ndimage.filters import gaussian_filter1d
import scipy.signal as signal
import numpy as np
import pandas as pd

# Misc utils for dealing with xarray structures:
def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    """1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """

    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def estimate_fs(da):
    sample_period = mode(np.diff(da.datetime.values), keepdims=True).mode[0]
    assert isinstance(sample_period, np.timedelta64)
    sample_period = sample_period / pd.to_timedelta(1, "s")
    return 1 / sample_period


def get_smoothed_da(da, smoothing_sigma=10, in_place=False):
    if not in_place:
        da = da.copy()
    da.values = gaussian_smooth(da, smoothing_sigma, estimate_fs(da))
    return da


def get_smoothed_ds(ds, smoothing_sigma=10, in_place=False):
    if not in_place:
        ds = ds.copy()
    for da_name, da in ds.items():
        ds[da_name] = get_smoothed_da(da, smoothing_sigma, in_place)
    return ds


def decimate(sig, q=5):
    dat = signal.decimate(sig.values, q=q, ftype="fir", axis=0)
    rs = xr.DataArray(
        dat,
        dims=sig.dims,
        coords={
            **sig["datetime"][::q].coords,
            **sig[sig.dims[-1]].coords,
        },
        attrs=sig.attrs,
    )
    rs.attrs["fs"] = sig.fs / q
    return rs
