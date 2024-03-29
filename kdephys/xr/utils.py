import tdt
import xarray as xr
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d, gaussian_filter
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

def gaussian_smooth_md(data, sigma, sampling_frequency=0.5, truncate=6):
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
    
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """

    return gaussian_filter(
        data, sigma * sampling_frequency, truncate=truncate, mode="constant"
    )


def estimate_fs(da):
    if 'datetime' in da.dims:
        sample_period = mode(np.diff(da.datetime.values), keepdims=True).mode[0]
        assert isinstance(sample_period, np.timedelta64)
        sample_period = sample_period / pd.to_timedelta(1, "s")
        return 1 / sample_period
    elif 'time' in da.dims:
        sample_period = mode(np.diff(da.time.values), keepdims=True).mode[0]
        #assert isinstance(sample_period, np.timedelta64)
        #sample_period = sample_period / pd.to_timedelta(1, "s")
        return 1 / sample_period
    else:
        raise ValueError("No time or datetime dimension found")


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


def rel_by_store(ds, state="NREM", t1=None, t2=None):
    """split a dataset by its stores, then get each store relative to a baseline recording (specified by t1 and t2), filtered by state.
    Relies on 'state' coordinate being up to date for best results.

    Args:
    -----------
        ds (xr.dataset, xr.DataArray): dataset
        state (str, optional): state to use in calculating the baseline average. Defaults to 'NREM'.
        t1 (pd.Timestamp, optional): start time of baseline. If not specified, 9am-9pm on the day of the first timestamp will be used. Defaults to None.
        t2 (pd.Timestamp, optional): end time of baseline. If not specified, 9am-9pm on the day of the first timestamp will be used. Defaults to None.
    Returns:
    -----------
        ds_rel (xr.dataset, xr.DataArray): dataset with each store relative to its baseline average.
    """
    ds_stores = {}
    if len(ds.prbs()) == 1:
        if t1 == None and t2 == None:
            rel_day = str(ds.datetime.values.min()).split("T")[0]
            t1 = pd.Timestamp(rel_day + " 09:00:00")
            t2 = pd.Timestamp(rel_day + " 21:00:00")
            avgs = ds.sel(datetime=slice(t1, t2)).st(state).mean("datetime")
            rel_ds = ds / avgs
            return rel_ds
        else:
            assert t1 != None and t2 != None, "must specify both t1 and t2"
            avgs = ds.sel(datetime=slice(t1, t2)).st(state).mean("datetime")
            rel_ds = ds / avgs
            return rel_ds
    elif len(ds.prbs()) > 1:
        for store in ds.prbs():
            ds_stores[store] = ds.prb(store)
        ds_rel = {}
        for store in ds_stores.keys():
            if t1 == None and t2 == None:
                rel_day = str(ds_stores[store].datetime.values.min()).split("T")[0]
                t1 = pd.Timestamp(rel_day + " 09:00:00")
                t2 = pd.Timestamp(rel_day + " 21:00:00")
                avg_vals = (
                    ds_stores[store]
                    .st(state)
                    .sel(datetime=slice(t1, t2))
                    .mean("datetime")
                )
                ds_rel[store] = ds.prb(store) / avg_vals
            else:
                assert t1 != None and t2 != None, "must specify both t1 and t2"
                avg_vals = (
                    ds_stores[store]
                    .st(state)
                    .sel(datetime=slice(t1, t2))
                    .mean("datetime")
                )
                ds_rel[store] = ds.prb(store) / avg_vals
        return xr.concat(ds_rel.values(), "store")
    else:
        print("no stores found")
        return None
