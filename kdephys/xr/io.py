import tdt
import xarray as xr
import numpy as np
import pandas as pd

##Functions for loading TDT SEV-stores
def sev_to_xarray(info, store):
    """Convert a single stream store to xarray format.

    Paramters:
    ----------
    info: tdt.StructType
        The `info` field of a tdt `blk` struct, as returned by `_load_stream_store`.
    store: tdt.StructType
        The store field of a tdt `blk.streams` struct, as returned by `_load_stream_store`.

    Returns:
    --------
    data: xr.DataArray (n_samples, n_channels)
        Values: The data, in microvolts.
        Attrs: units, fs
        Name: The store name
    """
    n_channels, n_samples = store.data.shape

    time = np.arange(0, n_samples) / store.fs + store.start_time
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta

    volts_to_microvolts = 1e6
    # had to add this try-except because stupid TDT defines 'channels' for EEG/LFP, but 'channel' for EMG.
    try:
        data = xr.DataArray(
            store.data.T * volts_to_microvolts,
            dims=("time", "channel"),
            coords={
                "time": time,
                "channel": store.channels,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            name=store.name,
        )
    except:
        data = xr.DataArray(
            store.data.T * volts_to_microvolts,
            dims=("time", "channel"),
            coords={
                "time": time,
                "channel": store.channel,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            name=store.name,
        )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data


def tev_to_xarray(info, store):
    """Convert a single stream store to xarray format.

    Paramters:
    ----------
    info: tdt.StructType
        The `info` field of a tdt `blk` struct, as returned by `_load_stream_store`.
    store: tdt.StructType
        The store field of a tdt `blk.streams` struct, as returned by `_load_stream_store`.

    Returns:
    --------
    data: xr.DataArray (n_samples, n_channels)
        Values: The data, in microvolts.
        Attrs: units, fs
        Name: The store name
    """
    n_channels, n_samples = store.data.shape

    time = np.arange(0, n_samples) / store.fs + store.start_time
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta

    volts_to_microvolts = 1e6
    data = xr.DataArray(
        store.data.T * volts_to_microvolts,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": store.channel,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        name=store.name,
    )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data


def load_tev_store(path, t1=0, t2=0, channel=None, store=""):

    data = tdt.read_block(path, channel=channel, store=store, t1=t1, t2=t2)
    store = data.streams[store]
    info = data.info
    datax = tev_to_xarray(info, store)
    return datax


def load_sev_store(path, t1=0, t2=0, channel=None, store=""):

    data = tdt.read_block(path, channel=channel, store=store, t1=t1, t2=t2)
    store = data.streams[store]
    info = data.info
    datax = sev_to_xarray(info, store)
    return datax


def get_data(
    block_path,
    store="",
    t1=0,
    t2=0,
    channel=None,
    sev=True,
    pandas=False,
    sel_chan=False,
):
    if sev == True:
        data = load_sev_store(block_path, t1=t1, t2=t2, channel=channel, store=store)
    else:
        data = load_tev_store(block_path, t1=t1, t2=t2, channel=channel, store=store)
    try:
        data = data.swap_dims({"time": "datetime"})
    except ValueError:
        print("Passing ValueError on dimension swap in get_data")
    if pandas == True:
        data = data.to_dataframe().drop(labels=["time", "timedelta"], axis=1)
    if sel_chan:
        data = data.sel(channel=sel_chan)
    return data
