import numpy as np
import pandas as pd
import xarray as xr
import kdephys.plot.main as kp
import kdephys.hypno as hp

from scipy.signal import spectrogram
from functools import partial
from multiprocessing import Pool

import kdephys.utils.spectral as sp
bands = sp.bands
"""
Functions taken directly from ecephys.signal.timefrequency to remove dependency on ecephys package
------------------
"""


# This function is taken directly from neurodsp.spectral.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def all_arrays_equal(iterator):
    """Check if all arrays in the iterator are equal."""
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


def trim_spectrogram(freqs, times, spg, f_range=None, t_range=None):
    """Extract a frequency or time range of interest from a spectrogram.
    Parameters
    ----------
    freqs : 1d array
        Frequency values for the spectrogram.
    times : 1d array
        Time values for the spectrogram.
    spg : 2d array
        Spectrogram, or time frequency representation of a signal.
        Formatted as [n_freqs, n_time_windows].
    f_range : list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range : list of [float, float]
        Time range to restrict to, as [t_low, t_high].
    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    times_ext : 1d array
        Extracted segment time values
    spg_ext : 2d array
        Extracted spectrogram values.
    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high,
    and time ranges >= t_low and <= t_high. It does not round to below
    or above f_low and f_high, or t_low and t_high, respectively.
    Examples
    --------
    Trim the spectrogram of a simulated time series:
    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.timefrequency import compute_wavelet_transform
    >>> from neurodsp.utils.data import create_times, create_freqs
    >>> fs = 500
    >>> n_seconds = 10
    >>> times = create_times(n_seconds, fs)
    >>> sig = sim_combined(n_seconds, fs,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs = create_freqs(1, 15)
    >>> mwt = compute_wavelet_transform(sig, fs, freqs)
    >>> spg = abs(mwt)**2
    >>> freqs_ext, times_ext, spg_ext = trim_spectrogram(freqs, times, spg,
    ...                                                  f_range=[8, 12], t_range=[0, 5])
    """

    # Initialize spg_ext, to define for case in which neither f_range nor t_range is defined
    spg_ext = spg

    # Restrict frequency range of the spectrogram
    if f_range is not None:
        f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
        freqs_ext = freqs[f_mask]
        spg_ext = spg_ext[f_mask, :]
    else:
        freqs_ext = freqs

    # Restrict time range of the spectrogram
    if t_range is not None:
        times_mask = np.logical_and(times >= t_range[0], times <= t_range[1])
        times_ext = times[times_mask]
        spg_ext = spg_ext[:, times_mask]
    else:
        times_ext = times

    return freqs_ext, times_ext, spg_ext


def check_spg_settings(fs, window, nperseg, noverlap):
    """Check settings used for calculating spectrogram.
    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int or None
        Length of each segment, in number of samples.
    noverlap : int or None
        Number of points to overlap between segments.
    Returns
    -------
    nperseg : int
        Length of each segment, in number of samples.
    noverlap : int
        Number of points to overlap between segments.
    """

    # Set the nperseg, if not provided
    if nperseg is None:
        # If the window is a string or tuple, defaults to 1 second of data
        if isinstance(window, (str, tuple)):
            nperseg = int(fs)
        # If the window is an array, defaults to window length
        else:
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    return nperseg, noverlap


def single_spectrogram_welch(
    sig,
    fs,
    window="hann",
    detrend="constant",
    nperseg=None,
    noverlap=None,
    f_range=None,
    t_range=None,
):
    """Compute spectrogram using Welch's method.

    Parameters
    -----------
    sig : (n_samples,)
        Time series.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    detrend: str or function or False, optional
        Specifies how to detrend each segment. If detrend is a string, it is passed as the
        type argument to the detrend function. If it is a function, it takes a segment and
        returns a detrended segment. If detrend is False, no detrending is done.
        Defaults to ‘constant’, which is mean subtraction.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 8.
    f_range: list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range: list of [float, float]
        Time range to restrict to, as [t_low, t_high].

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times)
        Spectrogram of `sig`.
    """

    # Calculate the short time Fourier transform with signal.spectrogram
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, spg_times, spg = spectrogram(
        sig, fs, window, nperseg, noverlap, detrend=detrend
    )
    freqs, spg_times, spg = trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    return freqs, spg_times, spg


def parallel_spectrogram_welch(sig, fs, **kwargs):
    """Apply `_compute_spectrogram_welch` to each channel in parallel.

    Should also work fine for a single channel, as long as sig is 2D.
    But in that case, maybe you want to save the overhead and use
    single_spectrogram_welch directly...

    Parameters
    ----------
    sig: (n_samples, n_chans)
        The multichannel timeseries.
    fs: float
        The sampling frequency of the data.
    **kwargs: optional
        Keyword arguments passed to `_compute_spectrogram_welch`.

    Returns:
    --------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times, n_chans)
        Spectrogram of `sig`.
    """

    worker = partial(single_spectrogram_welch, fs=fs, **kwargs)
    jobs = [x for x in sig.T]

    n_chans = sig.shape[1]
    with Pool(n_chans) as p:
        freqs, spg_times, spg = zip(*p.map(worker, jobs))

    assert all_arrays_equal(
        freqs
    ), "Spectrogram frequecies must match for all channels."
    assert all_arrays_equal(spg_times), "Segment times must match for all channels."

    freqs = freqs[0]
    spg_times = spg_times[0]
    if len(spg) > 1:
        spg = np.dstack(spg)
    else:
        spg = np.expand_dims(spg, axis=-1)

    return freqs, spg_times, spg


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

    if "channel" in sig.dims:
        freqs, spg_time, spg = parallel_spectrogram_welch(
            sig.transpose("time", "channel").values, sig.fs, **kwargs
        )
    else:
        print("Single channel spectrogram")
        freqs, spg_time, spg = single_spectrogram_welch(sig.values, sig.fs, **kwargs)

    time = sig.time.values.min() + spg_time
    
    if 'timedelta' in list(sig.coords):
        timedelta = sig.timedelta.values.min() + pd.to_timedelta(spg_time, "s")
    if 'datetime' in list(sig.coords):
        datetime = sig.datetime.values.min() + pd.to_timedelta(spg_time, "s")

    if "channel" in sig.dims:
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
    else:
        xarray_spg = xr.DataArray(
            spg,
            dims=("frequency", "time"),
            coords={
                "frequency": freqs,
                "time": time,
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
    hyp --> DateTimeHypnogram (defined in kdephys.ecephys_hypnogram)
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
    hyp --> DateTimeHypnogram (defined in kdephys.ecephys_hypnogram)
    state --> list of strings providing the states to KEEP
    """
    assert (
        "datetime" in bp_set.dims
    ), f"Need datetime dimension in bp_set, current dimensions are {bp_set.dims}"

    bp_set = bp_set.sel(
        datetime=hyp.keep_states(state).covers_time(bp_set.datetime)
    ).dropna(dim="datetime")

    return bp_set


def bp_melt(bp, bp_def=bands):
    """Melts a bandpower set to long-form.

    Parameters:
    -----------
    bp_def: bandpower dictionary, supplied automatically from kdephys.utils.spectral
    """

    bp_melt = pd.melt(
        bp,
        id_vars=["datetime", "channel", "store", "state"],
        value_vars=list(bp_def.keys()),
    )
    bp_melt.columns = ["datetime", "channel", "store", "state", "Band", "Bandpower"]
    return bp_melt


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
        DateTimeHypnogram (defined in kdephys.ecephys_hypnogram)
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


def get_muscle_energy(m, window_length=8, overlap=1):
    fs = m.fs
    m_data = m.values
    nperseg = int(window_length * fs)
    noverlap = int(overlap * fs)
    nstep = nperseg - noverlap
    shape = m_data.shape[:-1] + ((m_data.shape[-1] - noverlap) // nstep, nperseg)
    strides = m_data.strides[:-1] + (nstep * m_data.strides[-1], m_data.strides[-1])
    chunked_data = np.lib.stride_tricks.as_strided(m_data, shape=shape, strides=strides)

    energies = np.empty(0)
    for chunk in chunked_data:
        abs_chunk = np.absolute(chunk)
        energy_of_chunk = abs_chunk.mean()
        energies = np.append(energies, energy_of_chunk)

    return energies
