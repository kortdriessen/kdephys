import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # TODO remove seaborn dependency
import kdephys.xr.utils as xu
import kdephys.xr.spectral as xsp
from kdephys.utils.plots import hypno_colors


# This function is taken directly from neurodsp.plts.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def check_ax(ax, figsize=None):
    """Check whether a figure axes object is defined, define if not.
    Parameters
    ----------
    ax : matplotlib.Axes or None
        Axes object to check if is defined.
    Returns
    -------
    ax : matplotlib.Axes
        Figure axes object to use.
    """

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    return ax


def quick_lineplot(data):
    f, ax = plt.subplots(figsize=(35, 10))
    ax = sns.lineplot(x=data.datetime, y=data.values, ax=ax)
    return ax


def quick_histo(data):
    f, ax = plt.subplots(figsize=(10, 10))
    ax = sns.histplot(data=data, ax=ax, bins=1000)
    return ax


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
            alpha=0.25,
            color=hypno_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax

def add_light_schedule(times, ax=None, xlim=None):
    """add a bar to indicate light/dark periods at the top of an axes.

    Parameters
    ----------
    times: list
        border times, should start with first lights on.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    ax = check_ax(ax)
    for i, time in enumerate(times):
        if i == len(times)-1:
            break
        elif i % 2 == 0:
            ax.axvspan(time, times[i+1], ymin=0.98, ymax=1, color='gold')
        else:
            ax.axvspan(time, times[i+1], ymin=0.98, ymax=1, color='darkblue')

    ax.set_xlim(xlim)
    return ax

def mark_stim(start, end, ax=None, xlim=None, color='red'):
    """add a markings to an axis to indicate the start end end of a stimulus

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    ax = check_ax(ax)
    ax.axvline(start, color=color, linestyle='--', linewidth=2.2)
    ax.axvline(end, color=color, linestyle='--', linewidth=2.2)

    ax.set_xlim(xlim)
    return ax

def mark_single_stim(df, ax=None, xlim=None, color='red'):
    start = df.stm().datetime.min() if 'datetime' in df else df.stm().index.min()
    end = df.stm().datetime.max() if 'datetime' in df else df.stm().index.max()
    return mark_stim(start, end, ax, xlim, color)

def plot_shaded_bp(bp_set, chan, band, hyp, ax):

    bp = bp_set[band].sel(channel=chan)
    bp = xu.get_smoothed_da(bp, smoothing_sigma=6)

    ax = sns.lineplot(x=bp.datetime, y=bp, ax=ax)
    if hyp is not None:
        shade_hypno_for_me(hypnogram=hyp, ax=ax)
    ax.set(
        xlabel=None, ylabel="Raw " + band.capitalize() + " Power", xticks=[], xmargin=0
    )
    return ax


def spectro_plotter(
    spg,
    chan,
    f_range=slice(0, 35),
    t_range=None,
    yscale="linear",
    figsize=(35, 10),
    vmin=None,
    vmax=None,
    title="Title",
    ax=None,
):
    try:
        # spg = spg.swap_dims({'datetime': 'time'})
        spg = spg.sel(channel=chan, frequency=f_range)
    except IndexError:
        print("Already had time dimension - passing index error")

    freqs = spg.frequency
    spg_times = spg.datetime.values
    # freqs, spg_times, spg = dsps.trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    ax = check_ax(ax, figsize=figsize)
    im = ax.pcolormesh(
        spg_times,
        freqs,
        np.log10(spg),
        cmap="nipy_spectral",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
        shading="gouraud",
    )
    # ax.figure.colorbar(im)
    ax.set_yscale(yscale)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time")
    ax.set_title(title)

    if yscale == "log":
        ax.set_ylim(np.min(freqs[freqs > 0]), np.max(freqs))
    return ax


def plot_bps_all(spg, chans, ix, hyp, bp_def, band, fig_size=(35, 10)):
    fig, axes = plt.subplots(
        ncols=1, nrows=2 * len(chans), figsize=fig_size, sharex=True
    )
    ax_index = np.arange(0, len(chans))
    for chan, ux in zip(chans, ax_index):
        bax = ix[ux][0]
        sax = ix[ux][1]
        spg = spg.sel(channel=chan)
        plot_shaded_bp(spg, chan, bp_def, band, hyp, ax=axes[bax])
        spectro_plotter(spg, chan, ax=sax, figsize=(67, 10))


def plot_bp_and_spectro(spg, chan, hyp, bp_def, band, fig_size=(35, 10)):
    fig, (bx, sx) = plt.subplots(nrows=2, ncols=1, figsize=fig_size, sharex=True)
    bx = plot_shaded_bp(spg, chan, bp_def, band, hyp, ax=bx)
    sx = spectro_plotter(spg, chan, ax=sx, figsize=fig_size)
    return fig


def compare_psd(
    psd1,
    psd2,
    state,
    keys=["condition1", "condition2"],
    key_name="condition",
    scale="log",
    palette=["cyan", "white"],
):
    df = pd.concat(
        [psd1.to_dataframe("power"), psd2.to_dataframe("power")], keys=keys
    ).rename_axis(index={None: key_name})
    g = sns.relplot(
        data=df,
        x="frequency",
        y="power",
        hue=key_name,
        row="channel",
        kind="line",
        legend=False,
        aspect=(16 / 9),
        height=6,
        ci=None,
        palette=palette,
    )
    g.set(
        xscale=scale,
        yscale=scale,
        ylabel="NREM PSD, Normalized to Baseline",
        xlabel="Frequency",
        xlim=(1, 40),
    )
    return g


def plot_bp_set(bp_set, hyp, channel, ss=12, figsize=(30, 30), title=None):
    bp_set = bp_set.sel(channel=channel)
    bp_set = xu.get_smoothed_ds(bp_set, smoothing_sigma=ss)

    bands = list(bp_set.keys())
    ax_index = np.arange(0, len(list(bands)))

    fig, axes = plt.subplots(ncols=1, nrows=len(bands), figsize=figsize)

    for i, k in zip(ax_index, bands):
        fr = bp_set[k].f_range
        fr_str = "(" + str(fr[0]) + " -> " + str(fr[1]) + " Hz)"
        ax = sns.lineplot(x=bp_set[k].datetime, y=bp_set[k], ax=axes[i])
        ax.set_ylabel("Raw " + k.capitalize() + " Power")
        ax.set_title(k.capitalize() + " Bandpower " + fr_str)
    fig.suptitle(title)
    fig.tight_layout(pad=1.5)
    return fig, axes


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


def plot_muscle(
    m,
    mspg,
    bp_def=dict(
        super_deep=(0, 1),
        sub_delta=(0.5, 2),
        delta=(0.5, 4),
        theta=(4, 8),
        alpha=(8, 13),
        sigma=(11, 16),
        beta=(13, 30),
        low_gamma=(30, 55),
        high_gamma=(65, 90),
        omega=(100, 500),
    ),
    ss=12,
    title="",
):
    m = m.sel(channel=1)
    mspg = mspg.sel(channel=1)
    fig, axes = plt.subplots(ncols=1, nrows=len(bp_def) + 1, figsize=(30, 30))
    mbp = xsp.get_bp_set(mspg, bp_def)
    mbps = xsp.get_smoothed_ds(mbp, smoothing_sigma=ss)
    me = get_muscle_energy(m)

    ax_index = np.arange(0, len(bp_def))
    keys = list(bp_def.keys())
    for i, k in zip(ax_index, keys):
        fr = mbps[k].f_range
        fr_str = "(" + str(fr[0]) + " -> " + str(fr[1]) + " Hz)"
        ax = sns.lineplot(x=mbps[k].datetime, y=mbps[k], ax=axes[i])
        ax.set_ylabel("Raw " + k.capitalize() + " Power")
        ax.set_title(k.capitalize() + " Bandpower " + fr_str)
    ax = sns.lineplot(x=mbps.datetime.values, y=me, ax=axes[10])
    fig.suptitle(title)
    fig.tight_layout(pad=1.5)
    return fig, axes
