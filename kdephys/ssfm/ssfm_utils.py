import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kdephys.xr.utils as xu
import kdephys.xr.spectral as xsp
from ecephys.plot import check_ax


bp_def = dict(
    delta=(0.5, 4),
    theta=(4, 8),
    sigma=(11, 16),
    beta=(13, 30),
    low_gamma=(30, 55),
    high_gamma=(65, 90),
    omega=(300, 700),
)

hypno_colors = {
    "Wake": "forestgreen",
    "Brief-Arousal": "chartreuse",
    "Transition-to-NREM": "lightskyblue",
    "Transition-to-Wake": "palegreen",
    "NREM": "royalblue",
    "Transition-to-REM": "plum",
    "REM": "magenta",
    "Transition": "grey",
    "Art": "crimson",
    "Wake-art": "crimson",
    "Unsure": "white",
}


def build_hypno_for_me(states_and_times):
    """states_and_times --> series with only the timepoints corresponding sleep states (i.e. data that has already been 'scored')"""

    # Get boolean series for each state
    nrem_bool = states_and_times == "NREM"
    wake_bool = states_and_times == "Wake"
    rem_bool = states_and_times == "REM"

    # Use the boolean series to get start and end times for each state
    nrem_sne = starts_and_ends(nrem_bool)
    wake_sne = starts_and_ends(wake_bool)
    rem_sne = starts_and_ends(rem_bool)

    # Then we convert the start and end times for each state to a partial hypnogram
    nrem_hyp = pd.DataFrame(columns=["state", "end_time", "start_time", "duration"])
    nrem_hyp[["start_time", "end_time"]] = nrem_sne
    nrem_hyp["duration"] = nrem_hyp.end_time - nrem_hyp.start_time
    nrem_hyp["state"] = "NREM"

    wake_hyp = pd.DataFrame(columns=["state", "end_time", "start_time", "duration"])
    wake_hyp[["start_time", "end_time"]] = wake_sne
    wake_hyp["duration"] = wake_hyp.end_time - wake_hyp.start_time
    wake_hyp["state"] = "Wake"

    rem_hyp = pd.DataFrame(columns=["state", "end_time", "start_time", "duration"])
    rem_hyp[["start_time", "end_time"]] = rem_sne
    rem_hyp["duration"] = rem_hyp.end_time - rem_hyp.start_time
    rem_hyp["state"] = "REM"

    # Then we concat those and sort by the start_time
    fin_hypno = pd.concat([nrem_hyp, wake_hyp, rem_hyp])
    fin_hypno = fin_hypno.sort_values("start_time").reset_index(drop=True)

    return fin_hypno


def starts_and_ends(s, minimum_duration=np.timedelta64(3, "s")):
    start_times = np.empty(0)
    end_times = np.empty(0)
    period = s.index[1] - s.index[0]

    s_trues = s[s == True]
    ix = s_trues.index
    ix_counter = np.arange(0, len(ix))

    try:
        for i in ix_counter:
            if (ix[i] - period) != ix[i - 1]:
                start_times = np.append(start_times, ix[i])
            if (ix[i] + period) != ix[i + 1]:
                end_times = np.append(end_times, (ix[i] + period))
            elif np.logical_and(
                (ix[i] + period) == ix[i + 1], (ix[i] - period) == ix[i - 1]
            ):
                pass
    except IndexError:
        print("passing indexing error (starts_and_ends)")
        pass
    end_times = np.append(end_times, (ix[ix_counter.max()] + period))
    return [
        (start_time, end_time)
        for start_time, end_time in zip(start_times, end_times)
        if end_time >= (start_time + minimum_duration)
    ]


"""
PLOTTING FUNCTIONS
------------------
"""


def threshplot(data, time=None, percentiles=[50, 60, 65, 70], cut=None):
    f, h_ax = plt.subplots(figsize=(20, 10))
    if cut:
        lim = np.percentile(data, cut)
        data = data[data < cut]
        h_ax.set_xlim(0, lim)
    h_ax = sns.histplot(data=data, ax=h_ax, bins=50)
    h_ax.axvline(np.percentile(data, percentiles[0]), color="magenta")
    h_ax.axvline(np.percentile(data, percentiles[1]), color="b")
    h_ax.axvline(np.percentile(data, percentiles[2]), color="forestgreen")
    h_ax.axvline(np.percentile(data, percentiles[3]), color="r")
    plt.show()

    f, lin_ax = plt.subplots(figsize=(40, 10))
    lin_ax = sns.lineplot(x=time, y=data, ax=lin_ax)
    lin_ax.axhline(np.percentile(data, percentiles[0]), color="magenta")
    lin_ax.axhline(np.percentile(data, percentiles[1]), color="b")
    lin_ax.axhline(np.percentile(data, percentiles[2]), color="forestgreen")
    lin_ax.axhline(np.percentile(data, percentiles[3]), color="r")
    if cut:
        lim = np.percentile(data, cut)
        lin_ax.set_ylim(0, lim)
    plt.show()
    return h_ax, lin_ax


def plot_hypno_for_me(hypno, spg, emg_spg, bp_def=bp_def, chan=2, smooth=True):
    fig, (m, d, g) = plt.subplots(ncols=1, nrows=3, figsize=(35, 15))
    try:
        emg_spg = emg_spg.sel(channel=1)
    except KeyError:
        emg_spg = emg_spg
    try:
        spg = spg.sel(channel=chan)
    except KeyError:
        spg = spg

    # plot muscle activity
    emg_bp = xsp.get_bandpower(emg_spg, bp_def["omega"])
    if smooth == True:
        emg_bp = xu.get_smoothed_da(emg_bp, smoothing_sigma=14)
    sns.lineplot(x=spg.datetime, y=emg_bp, color="black", ax=m)
    shade_hypno_for_me(hypnogram=hypno, ax=m)
    m.set_title("Muscle Activity (Full Spectrum)")

    # plot delta power
    delta = xsp.get_bandpower(spg, bp_def["delta"])
    if smooth == True:
        delta = xu.get_smoothed_da(delta, smoothing_sigma=14)
    sns.lineplot(x=delta.datetime, y=delta, color="black", ax=d)
    shade_hypno_for_me(hypnogram=hypno, ax=d)
    d.set_title("EEG-" + str(chan) + " Delta Bandpower")

    # plot gamma power
    gamma = xsp.get_bandpower(spg, bp_def["high_gamma"])
    if smooth == True:
        gamma = xu.get_smoothed_da(gamma, smoothing_sigma=14)
    sns.lineplot(x=spg.datetime, y=gamma, color="black", ax=g)
    shade_hypno_for_me(hypnogram=hypno, ax=g)
    g.set_title("EEG-" + str(chan) + " Gamma Bandpower")

    return m, d, g


def plot_hypno_for_me_v4(hypno, spg, emg_energy, bp_def, chan=2, smooth=True):
    fig, (m, d, g) = plt.subplots(ncols=1, nrows=3, figsize=(20, 10))
    try:
        spg = spg.sel(channel=chan)
    except KeyError:
        spg = spg

    # plot muscle activity
    sns.lineplot(x=spg.datetime, y=emg_energy, color="black", ax=m)
    shade_hypno_for_me(hypnogram=hypno, ax=m)
    m.set_title("Muscle Activity (Absolute Value of Amplitude)")

    # plot delta power
    delta = xsp.get_bandpower(spg, bp_def["delta"])
    if smooth == True:
        delta = xu.get_smoothed_da(delta, smoothing_sigma=8)
    sns.lineplot(x=delta.datetime, y=delta, color="black", ax=d)
    shade_hypno_for_me(hypnogram=hypno, ax=d)
    d.set_title("EEG-" + str(chan) + " Delta Bandpower")
    d.set_ylim(0, 10000)

    # plot gamma power
    gamma = xsp.get_bandpower(spg, bp_def["gamma"])
    if smooth == True:
        gamma = xu.get_smoothed_da(gamma, smoothing_sigma=10)
    sns.lineplot(x=spg.datetime, y=gamma, color="black", ax=g)
    shade_hypno_for_me(hypnogram=hypno, ax=g)
    g.set_title("EEG-" + str(chan) + " Gamma Bandpower")

    return fig


def shade_hypno_for_me(hypnogram, ax=None, xlim=None):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    ax = check_ax(ax)
    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            alpha=0.5,
            color=hypno_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax


def get_frac_oc(hyps, hyp_keys):
    df = (
        pd.concat(
            [hyp.fractional_occupancy().to_frame() for hyp in hyps], keys=hyp_keys
        ).unstack()
        * 100
    )
    return df


def compare_hypnos_for_me(spg, chan, ssfm_hyp, your_hyp, smooth=True):
    try:
        spg = spg.sel(channel=chan)
    except KeyError:
        spg = spg
    f, (ssfm_hyp_ax, your_hyp_ax) = plt.subplots(nrows=2, ncols=1, figsize=(35, 15))
    spg = xsp.get_bandpower(spg, (0.5, 4))
    if smooth == True:
        spg = xu.get_smoothed_da(spg, smoothing_sigma=10)
    ssfm_hyp_ax = sns.lineplot(x=spg.datetime, y=spg, ax=ssfm_hyp_ax)
    ssfm_hyp_ax.set_title("SSFM Hypnogram")
    your_hyp_ax = sns.lineplot(x=spg.datetime, y=spg, ax=your_hyp_ax)
    your_hyp_ax.set_title("Your Hypnogram")
    shade_hypno_for_me(ssfm_hyp, ax=ssfm_hyp_ax)
    shade_hypno_for_me(your_hyp, ax=your_hyp_ax)
    fo = get_frac_oc([ssfm_hyp, your_hyp], ["SSFM", "Human"])
    return fo
