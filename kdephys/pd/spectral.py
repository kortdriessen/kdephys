from kdephys.pd.ecdata import ecdata
import numpy as np
import pandas as pd
import kdephys.utils.spectral as sp
import ecephys.signal.timefrequency as tfr
import xarray as xr

bands = sp.bands


def pd_spg(df, window_length=4, overlap=2, **kwargs):

    # Get the raw data from the dataframe:
    d = df.drop(["tdt_time", "timedelta", "datetime"], axis=1, inplace=False)
    raw_data = d.to_numpy()
    fs = df.fs
    chans = list(d)
    chans_int = [int(i) for i in chans]

    # Compute the ecdata powergram:
    kwargs["nperseg"] = int(window_length * fs)  # window length in number of samples
    kwargs["noverlap"] = int(overlap * fs)  # overlap in number of samples
    kwargs["f_range"] = [0, 100]  # frequency range to compute the spectrogram
    freqs, spg_time, spg = tfr.parallel_spectrogram_welch(raw_data, fs, **kwargs)
    tdt_time = df["tdt_time"].min() + spg_time
    timedelta = df["timedelta"].min() + pd.to_timedelta(spg_time, "s")
    datetime = df["datetime"].min() + pd.to_timedelta(spg_time, "s")
    xrda = xr.DataArray(
        spg,
        dims=("frequency", "datetime", "channel"),
        coords={
            "frequency": freqs,
            "datetime": datetime,
            "channel": chans_int,
            "timedelta": ("datetime", timedelta),
            "tdt_time": ("datetime", tdt_time),
        },
    )
    return ecdata(xrda.to_dataframe(name="spg").reset_index())


def pd_bp(spg_df, band_dict=bands):
    "expects a spectrogram dataframe from pd_spg"
    td_ix = np.repeat(pd.unique(spg_df["timedelta"]), len(pd.unique(spg_df["channel"])))
    spg_df = spg_df.set_index(["frequency", "datetime", "channel"])

    # Create a new dataframe to hold the bandpower data:
    bp = spg_df.xs(slice(1, 2), level="frequency", drop_level=False)
    bp = bp.groupby(level=["datetime", "channel"]).sum()
    bp_df = pd.DataFrame()

    # bp_df[['datetime', 'channel']] = spg_df[['datetime', 'channel']]

    # Calculate the power in each band:
    for band in band_dict:
        bp = spg_df.xs(band_dict[band], level="frequency", drop_level=False)
        bp = bp.groupby(level=["datetime", "channel"]).sum()
        bp_df[band] = bp["spg"]

    bp_df["timedelta"] = td_ix
    return ecdata(bp_df.reset_index())
