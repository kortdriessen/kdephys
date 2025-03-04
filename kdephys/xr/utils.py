import tdt
import xarray as xr
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import scipy.signal as signal
import numpy as np
import pandas as pd
import zarr
import dask


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
    if 'channel' in sig.dims:
        rs = xr.DataArray(
            dat,
            dims=sig.dims,
            coords={
                **sig["datetime"][::q].coords,
                **sig[sig.dims[-1]].coords,
            },
            attrs=sig.attrs,
        )
    else:
        rs = xr.DataArray(
            dat,
            dims=sig.dims,
            coords={
                **sig["datetime"][::q].coords,
            },
            attrs=sig.attrs,
        )
    rs.attrs["fs"] = sig.fs / q
    return rs

def rel_by_store(ds, state="NREM", t1=None, t2=None, method='mean'):
    """split a dataset by its stores, then get each store relative to a 
    baseline recording (specified by t1 and t2), filtered by state.
    
    - Relies on 'state' coordinate being up to date for best results.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing both stores and the baseline data.
    state : str, optional
        state in which to compute the baseline value, by default "NREM"
    t1 : pd.Timestamp, optional
        start time of baseline. If not specified, 9am on the day of the first timestamp will be used, by default None
    t2 : pd.Timestamp, optional
        end time of baseline. If not specified, 9pm on the day of the first timestamp will be used, by default None
    method : str, optional
        whether to reference to the baseline mean or median, by default 'mean'
    """
    if method == 'mean':
        return rel_by_store_mean(ds, state=state, t1=t1, t2=t2)
    elif method == 'median':
        return rel_by_store_median(ds, state=state, t1=t1, t2=t2)
    else:
        raise ValueError(f"method parameter '{method}' not recognized, must use 'mean' or 'median'")

def rel_by_store_mean(ds, state="NREM", t1=None, t2=None):
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

def rel_by_store_median(ds, state="NREM", t1=None, t2=None):
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
            avgs = ds.sel(datetime=slice(t1, t2)).st(state).median("datetime")
            rel_ds = ds / avgs
            return rel_ds
        else:
            assert t1 != None and t2 != None, "must specify both t1 and t2"
            avgs = ds.sel(datetime=slice(t1, t2)).st(state).median("datetime")
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
                    .median("datetime")
                )
                ds_rel[store] = ds.prb(store) / avg_vals
            else:
                assert t1 != None and t2 != None, "must specify both t1 and t2"
                avg_vals = (
                    ds_stores[store]
                    .st(state)
                    .sel(datetime=slice(t1, t2))
                    .median("datetime")
                )
                ds_rel[store] = ds.prb(store) / avg_vals
        return xr.concat(ds_rel.values(), "store")
    else:
        print("no stores found")
        return None


def _get_increasing_segments_mask(times):
    keep = np.ones((len(times),), dtype=bool)
    gap_ixs = np.where(np.diff(times) < 0)[0]
    for gap_ix in gap_ixs:
        pre_gap_values = times[gap_ix]
        next_ix = np.where(times[gap_ix:] > pre_gap_values)[0][0]
        keep[gap_ix+1:gap_ix+next_ix] = False
    return keep

def _hotfix_times(da):
    keep = _get_increasing_segments_mask(da.time.data)
    return da.sel(time=keep).copy()

def load_raw_processed_zarr(path):
    zg = zarr.open(path)
    return zg

def load_processed_zarr_as_xarray(fpath, times=False):
    """Load SI-saved zarr as dask-based xarray for OFF detection.
    
    NB: Non monotonously increasing timestamps are dismissed."""

    zg = zarr.open(fpath)
    print('zarr opened')
    times = zg.times_seg0 if times else np.linspace(0, zg.traces_seg0.shape[0] / zg.attrs["sampling_frequency"], zg.traces_seg0.shape[0])
    da = xr.DataArray(
        data=dask.array.from_zarr(zg.traces_seg0).rechunk(),
        dims=("time", "channel"),
        coords={"time": times, "channel": np.arange(1, 17)},
        attrs = {
            "units": "mV",
            "fs": zg.attrs["sampling_frequency"],
        },
        name="processed_mua",
    )
    print('da created')
    #if times:
    #    times = zg.times_seg0
    #else:
    #    duration = zg.traces_seg0.shape[0] / zg.attrs["sampling_frequency"]
    #    times = np.linspace(0, duration, da.time.shape[0]) 
#
    #channels = np.arange(1, 17)
    #da = da.assign_coords({'time':times, 'channel':channels})
    
    #print('hotfixing times')
    #da = _hotfix_times(da)

    return da

def xr_from_zarr_store(fpath, times=False):
    return