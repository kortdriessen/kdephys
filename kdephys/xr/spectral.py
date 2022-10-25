import numpy as np
import pandas as pd
import xarray as xr
import kdephys.plot.plot as kp
import ecephys as ec
import kdephys.hypno as hp
import kdephys.utils.spectral as sp

bands = sp.bands

"""
SPECTROGRAMS
------------------
"""


def get_spextrogram(
    sig, window_length=4, overlap=2, window="hann", f_range=None, t_range=None, **kwargs
):
    """Calculates a spectrogram and returns as xr.DataArray with dimensions datetime, frequency, channel

    Parameters
    ----------
    Sig --> Should be an xr.DataArray with time or datetime dimension, and a fs attribute
    see ecephys.signal.timefrequency.single_spectrogram_welch for details on kwargs
    """
    try:
        sig = sig.swap_dims({"datetime": "time"})
    except:
        print(
            "Passing Error in get_spextrogram because xarray already has time dimension, no need to swap it in"
        )

    # Add the kwargs
    # window length in number of samples
    kwargs["nperseg"] = int(window_length * sig.fs)

    # overlap in number of samples
    kwargs["noverlap"] = int(overlap * sig.fs)

    # window function
    kwargs["window"] = window

    # frequency range
    kwargs["f_range"] = f_range

    # time range
    kwargs["t_range"] = t_range

    freqs, spg_time, spg = ec.signal.timefrequency.parallel_spectrogram_welch(
        sig.transpose("time", "channel").values, sig.fs, **kwargs
    )

    time = sig.time.values.min() + spg_time
    timedelta = sig.timedelta.values.min() + pd.to_timedelta(spg_time, "s")
    datetime = sig.datetime.values.min() + pd.to_timedelta(spg_time, "s")

    xarray_spg = xr.DataArray(
        spg,
        dims=("frequency", "time", "channel"),
        coords={
            "frequency": freqs,
            "time": time,
            "channel": sig.channel.values,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        attrs={"units": f"{sig.units}^2/Hz"},
    )

    # return xarray_spg with default dimension = datetime
    return xarray_spg.swap_dims({"time": "datetime"})


def filt_spg_by_state(spg, hyp, state):
    """
    Provides a spectrogram filtered by a given state or set of states, provided a hypnogram
    spg --> xr.DataArray, Needs datetime dimension
    hyp --> DateTimeHypnogram (defined in ecephys.hypnogram)
    state --> list of strings specifying the states to KEEP
    """
    assert type(spg) == xr.core.dataarray.DataArray, "spg must be an xarray.DataArray"
    assert (
        "datetime" in spg.dims
    ), f"spg must have datetime dimension, current dimensions are: {spg.dims}"

    spg = spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime)).dropna(
        dim="datetime"
    )
    return spg


def get_spg_from_dataset(ds, window_length=4, overlap=2):
    spg_set = {}
    kl = list(ds.keys())
    for key in kl:
        spg_set[key] = get_spextrogram(ds[key], window_length, overlap)
    return spg_set


"""
BANDPOWER
--------------------
"""


def get_bandpower(spg, f_range):
    """Get band-limited power from a spectrogram.
    Parameters
    ----------
    spg: xr.DataArray (frequency, datetime, [channel])
        Spectrogram data.
    f_range: slice
        Frequency range to restrict to, as slice(f_low, f_high).
    Returns:
    --------
    bandpower: xr.DataArray (datetime, [channel])
        Sum of the power in `f_range` at each point in time.
    """
    bandpower = spg.sel(frequency=slice(*f_range)).sum(dim="frequency")
    bandpower.attrs["f_range"] = f_range

    return bandpower


def get_relative_bp(spg1, spg2, f_range=(0.5, 4), median=False):
    """
    Takes two spectrograms and returns a given bandpower for one of them (spg1) relative to the given bandpower of the other (spg2)

    No state-based filtering is done here, but state-specific spectrograms can be passed in

    The mean of spg2-bandpower is used as the default reference, unless median=True, in which case the median is used
    """

    bp1 = get_bandpower(spg1, f_range)
    bp2 = get_bandpower(spg2, f_range)

    if median == True:
        bp2 = bp2.median(dim="datetime")
    else:
        bp2 = bp2.mean(dim="datetime")

    return (bp1 / bp2) * 100


def get_bp_set(spg, bands=bands):
    """
    Returns a set of bandpower timeseries for a given spectrogram
    -------------------------------------------------------------
    spg --> xarray.DataArray with datetime dimension
    bands --> dictionary of frequency ranges
    """
    assert type(spg) == xr.core.dataarray.DataArray, "spg must be an xarray.DataArray"

    bp_ds = xr.Dataset({})
    bp_vars = {}
    keys = list(bands.keys())

    for k in keys:
        bp_vars[k] = get_bandpower(spg, bands[k])

    bp_set = bp_ds.assign(**bp_vars)

    return bp_set


def get_bp_from_dataset(ds):
    bp_set = {}
    kl = list(ds.keys())
    for key in kl:
        bp_set[key] = get_bp_set(ds[key])
    return bp_set


def filt_bp_set_by_state(bp_set, hyp, state):
    """
    Takes bandpower set with datetime dimension, filters by a given state in hyp
    bp_set --> xr.Dataset with datetime dimension
    hyp --> DateTimeHypnogram (defined in ecephys.hypnogram)
    state --> list of strings providing the states to KEEP
    """
    assert (
        "datetime" in bp_set.dims
    ), f"Need datetime dimension in bp_set, current dimensions are {bp_set.dims}"

    bp_set = bp_set.sel(
        datetime=hyp.keep_states(state).covers_time(bp_set.datetime)
    ).dropna(dim="datetime")

    return bp_set


"""
POWER SPECTRAL DENSITY
-----------------------
"""


def get_ss_psd(spg, hyp, state, median=True):
    """Provides a PSD of a given spectrogram, filtered by a given state or set of states in hyp

    Parameters
    ------------------------------------------------------------
    spg:
        Dataarray with datetime dimension
    hyp:
        DateTimeHypnogram (defined in ecephys.hypnogram)
    state:
        list of strings specifying the states to KEEP
    median:
        boolean, if True, returns median of the PSD, if False, returns mean
    """
    if median == True:
        return filt_spg_by_state(spg, hyp, state).median(dim="datetime")
    else:
        return filt_spg_by_state(spg, hyp, state).mean(dim="datetime")


def compare_psd(spg1, spg2, hyp1, hyp2, channel, keys):
    def compare_state_psd(
        state, scale="log", freqs=slice(None), channel=channel, keys=keys
    ):
        psd1 = hp.keep_states(spg1, hyp1, state).sel(frequency=freqs, channel=channel)
        psd2 = hp.keep_states(spg2, hyp2, state).sel(frequency=freqs, channel=channel)
        return kp.compare_psd(psd1, psd2, state=state, keys=keys, scale=scale)

    g = compare_state_psd(["NREM"])
    g = compare_state_psd(["Wake"])

    g = compare_state_psd(["NREM"], scale="linear", freqs=slice(0, 40))
    g = compare_state_psd(["Wake"], scale="linear", freqs=slice(0, 40))


def get_psd_rel2bl(bl_spg, exp_spg, bl_hyp, exp_hyp, state, chan, median=True):
    bl_psd = get_ss_psd(bl_spg, bl_hyp, state, median=median)
    exp_psd = get_ss_psd(exp_spg, exp_hyp, state, median=median)
    return (exp_psd / bl_psd * 100).sel(channel=chan)
