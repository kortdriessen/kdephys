import numpy as np
import pandas as pd
from . import ssfm_utils as ssu
from scipy.signal import butter, lfilter
import ecephys.hypnogram as hp
import kdephys.xr.spectral as xsp
import kdephys.xr.utils as xu
import xarray as xr


def get_bp_set2(spg, bands, pandas=False):
    if type(spg) == xr.core.dataset.Dataset:
        spg = spg.to_array(dim="channel")

    bp_ds = xr.Dataset({})
    bp_vars = {}
    keys = list(bands.keys())
    for k in keys:
        bp_vars[k] = xsp.get_bandpower(spg, bands[k])
    bp_set = bp_ds.assign(**bp_vars)
    if pandas == True:
        bp_set = bp_set.to_dataframe().drop(labels=["time", "timedelta"], axis=1)
        return bp_set
    else:
        return bp_set


bp_def_v4 = dict(
    sub_delta=slice(0, 1),
    delta=slice(0.5, 5),
    theta=slice(6, 9),
    alpha=slice(10.5, 15),
    sigma=slice(11, 16),
    beta=slice(22, 30),
    gamma=slice(35, 45),
    wide=slice(0, 30),
)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_muscle_energy(m, window_length=2, overlap=0, filt=True):
    fs = m.fs
    if filt == True:
        m_data = butter_bandpass_filter(m.values, 10, 100, fs, order=5)
    else:
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


def get_bp_features(eeg, bp_def, window_length=2, overlap=0, chan=2):
    spg = xsp.get_spextrogram(eeg, window_length=window_length, overlap=overlap)
    spg = spg.sel(channel=chan)
    bp_set = xsp.get_bp_set(spg, bp_def)
    return bp_set, spg


def get_norm_feature(f):
    l = np.percentile(f, [0, 10])
    h = np.percentile(f, [90, 100])

    f_low = f[np.where(np.logical_and(f >= l[0], f <= l[1]))]
    f_high = f[np.where(np.logical_and(f >= h[0], f <= h[1]))]
    fmin = f_low.mean()
    fmax = f_high.mean()
    fnorm = np.empty(0)
    for i in f:
        inorm = (i - fmin) / (fmax - fmin)
        fnorm = np.append(fnorm, inorm)
    tooneg = np.where(fnorm < 0)
    toopos = np.where(fnorm > 1)
    fnorm[tooneg] = 0
    fnorm[toopos] = 1
    return fnorm


def get_norm_bpset(bp):
    kl = list(bp.keys())
    for key in kl:
        f = bp[key].values
        fnorm = get_norm_feature(f)
        bp[key] = (["datetime"], fnorm)
    return bp


def get_indexes(bp, mus):
    """where bp_set is xr.DataSet, and mus is nd_array, with same number of data points"""
    # bp = get_norm_bpset(bp)
    # mus = get_norm_feature(mus)

    # wake index
    ixw = (mus * bp.gamma.values) / bp.delta.values
    ixw = get_norm_feature(ixw)

    # NREM Index
    ixn = (bp.delta.values * bp.alpha.values) / np.square(bp.gamma.values)
    ixn = get_norm_feature(ixn)

    # REM index
    ixr = np.power(bp.theta.values, 3) / (bp.delta.values * bp.alpha.values * mus)
    ixr = get_norm_feature(ixr)

    # Artifact Index
    ixa = (((2 * bp.sub_delta.values) + bp.beta.values) * bp.gamma.values) / (
        bp.delta.values
        + bp.theta.values
        + bp.alpha.values
        + bp.beta.values
        + bp.gamma.values
    )
    ixa = get_norm_feature(ixa)

    # delta index used for scoring out delta arts during wake:
    ixd = bp.delta.values
    ixd = get_norm_feature(ixd)

    # Muscle Index for general use
    ixm = mus
    ixm = get_norm_feature(ixm)

    # Bandpower ratios
    bpr_low = bp.sub_delta.values / bp.wide.values
    bpr_low = get_norm_feature(bpr_low)

    dt_ix = bp.datetime.values
    ix_df = pd.Series(ixw, index=dt_ix)
    ix_df = ix_df.to_frame(name="IXW")
    ix_df["IXN"] = ixn
    ix_df["IXR"] = ixr
    ix_df["IXA"] = ixa
    ix_df["PRL"] = bpr_low
    ix_df["IXD"] = ixd
    ix_df["IXM"] = ixm
    return ix_df


def average_indices(ix_df, nrows):
    ixr = np.arange(0, len(ix_df.index) + nrows, nrows)
    itr = np.arange(0, len(ixr) - 1)
    new_dt_ix = np.empty(0)
    avg_df = pd.DataFrame(columns=["IXW", "IXN", "IXR", "IXA"])
    for i in itr:
        chunk = ix_df.iloc[ixr[i] : ixr[i + 1]]
        dt = chunk.index[0]
        new_dt_ix = np.append(new_dt_ix, dt)

        mean = chunk.mean()
        avg_df = avg_df.append(mean, ignore_index=True)
    avg_df["dt_ix"] = new_dt_ix
    avg_df = avg_df.set_index("dt_ix")
    return avg_df


def scoring_decision_tree(x):
    x["state"] = np.NaN

    # decision-point #1
    x.loc[x.IXA / sum([x.IXW, x.IXN, x.IXR]) > 0.9, "state"] = "Wake"
    x.loc[
        np.logical_and(
            ((x.IXW > x.IXN) & (x.IXW > x.IXR)),
            np.logical_and(x.state != "REM", x.state != "NREM"),
        ),
        "state",
    ] = "Wake"
    x.loc[
        np.logical_and(
            ((x.IXN > x.IXW) & (x.IXN > x.IXR)),
            np.logical_and(x.state != "REM", x.state != "Wake"),
        ),
        "state",
    ] = "NREM"
    x.loc[
        np.logical_and(
            ((x.IXR > x.IXN) & (x.IXR > x.IXW)),
            np.logical_and(x.state != "NREM", x.state != "Wake"),
        ),
        "state",
    ] = "REM"

    # x.loc[np.logical_and(x.IXD>0.5, x.state=='Wake'),'state'] = 'Art'

    hypno = ssu.build_hypno_for_me(x["state"])
    return hp.DatetimeHypnogram(hypno)


def ssfm_v4(
    eeg,
    emg,
    chan,
    window_length=4,
    overlap=2,
    bp_def=bp_def_v4,
    avg=True,
    nrows=4,
    user_hyp=None,
):
    bp, eeg_spg = get_bp_features(
        eeg, bp_def, window_length=window_length, overlap=overlap, chan=chan
    )
    mus = get_muscle_energy(
        emg, window_length=window_length, overlap=overlap, filt=True
    )

    if len(bp.delta.values) != len(mus):
        print(len(bp.datetime.values))
        print(len(mus))
        # bp = bp.drop_isel(datetime=4799)
        # eeg_spg = eeg_spg.drop_isel(datetime=4799)

    ix_df = get_indexes(bp, mus)

    if avg == True:
        ix_df = average_indices(ix_df, nrows=nrows)

    hypno = scoring_decision_tree(ix_df)
    fig = ssu.plot_hypno_for_me_v4(hypno, eeg_spg, mus, bp_def, chan=chan)

    if user_hyp is not None:
        fo = ssu.compare_hypnos_for_me(eeg_spg, chan, hypno, user_hyp)
        return hp.DatetimeHypnogram(hypno), fo
    else:
        return hp.DatetimeHypnogram(hypno), fig
