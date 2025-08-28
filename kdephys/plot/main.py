from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # TODO remove seaborn dependency
import kdephys.xr.utils as xu
import kdephys.xr.spectral as xsp
from kdephys.utils.plots import hypno_colors
import polars as pl
from xhistogram.xarray import histogram


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


def shade_hypno_for_me(hypnogram, ax=None, xlim=None, ymin=0, ymax=1, alpha=0.15):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    from kdephys.utils.plots import hypno_colors
    xlim = ax.get_xlim() if (ax and not xlim) else xlim
    
    ax = check_ax(ax)
    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            ymin=ymin,
            ymax=ymax,
            alpha=alpha,
            color=hypno_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax

def _add_hspan(ax, xmin, xmax, lower=0.8, upper=0.9):
    ylim = ax.get_ylim()
    ax.axvspan(xmin, xmax, ymin=lower, ymax=upper, color='blue')
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
        if i == len(times) - 1:
            break
        elif i % 2 == 0:
            ax.axvspan(time, times[i + 1], ymin=0.98, ymax=1, color="gold")
        else:
            ax.axvspan(time, times[i + 1], ymin=0.98, ymax=1, color="darkblue")

    ax.set_xlim(xlim)
    return ax


def mark_stim(start, end, ax=None, xlim=None, color="red"):
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
    ax.axvline(start, color=color, linestyle="--", linewidth=2.2)
    ax.axvline(end, color=color, linestyle="--", linewidth=2.2)

    ax.set_xlim(xlim)
    return ax


def mark_single_stim(df, ax=None, xlim=None, color="red"):
    start = df.stm().datetime.min() if "datetime" in df else df.stm().index.min()
    end = df.stm().datetime.max() if "datetime" in df else df.stm().index.max()
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
    chan=None,
    f_range=None,
    t_range=None,
    yscale="linear",
    figsize=(35, 10),
    vmin=None,
    vmax=None,
    title="Title",
    ax=None,
):
    if f_range != None:
        spg = spg.sel(frequency=f_range)
    
    try:
        # spg = spg.swap_dims({'datetime': 'time'})
        spg = spg.sel(channel=chan)
    except:
        print("Passing error - no channel dimension")

    freqs = spg.frequency
    spg_times = spg.datetime.values if "datetime" in spg else spg.time.values
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


def gen_xr_plotter(da, ax, hyp, color=None):
    """A general function to plot xarray dataarrays, with hypnogram shading and the addition of light schedules.

    Parameters
    ----------
    da : _type_
        _description_
    ax : _type_
        _description_
    hyp : _type_
        _description_
    color : _type_, optional
        _description_, by default None
    """
    if color == None:
        ax.plot(da.datetime, da)
    else:
        ax.plot(da.datetime, da, color=color, linewidth=6)
    if hyp is not None:
        shade_hypno_for_me(hyp, ax)
    add_light_schedule(da.light_schedule(), ax)
    return ax


def bp_plot(bp, ax, hyp=None, color=None):
    if color == None:
        ax.plot(bp.datetime, bp)
    else:
        ax.plot(bp.datetime, bp, color=color, linewidth=8)
    if hyp is not None:
        shade_hypno_for_me(hyp, ax)
    add_light_schedule(bp.light_schedule(), ax)
    return ax


def _title(ax, subject, exp, extra=""):
    ax.set_title(f"{subject} | {exp} | {extra}", fontsize=18, fontweight="bold")
    return ax


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


def quick_bp_channel_plot(bp, band="delta"):
    for store in bp.store.values:
        for chan in bp.channel.values:
            f, ax = plt.subplots()
            bp[band].sel(store=store, channel=chan).plot(ax=ax)
    return


def _plot_overlapped_fp(data, df_map=None, pal=None, shade_df=None, hspace=-0.5, height=3, aspect=12):
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    if pal is None:
        pal = sns.cubehelix_palette(16, rot=-.25, light=.7)
    xname = 'time' if 'time' in data.columns else 't'
    if xname not in data.columns:
        xname = 'datetime'
    yname = 'data' if 'data' in data.columns else 'd'
    assert xname in data.columns, f"xname {xname} not in data"
    assert yname in data.columns, f"yname {yname} not in data"
    if 'condition' in data.columns:
        g = sns.relplot(data=data, clip_on=False, x=xname, y=yname, hue='condition', palette=pal, row='channel', kind='line', linewidth=3, aspect=aspect, height=height)
    else:
        g = sns.relplot(data=data, clip_on=False, x=xname, y=yname, palette=pal, hue='channel', row='channel', kind='line', linewidth=3, aspect=aspect, height=height)
    g.figure.subplots_adjust(hspace=hspace)


    if shade_df is not None:
        for channel in shade_df.channel.unique():
            shade_df_chan = shade_df.loc[shade_df.channel == channel]
            for row in shade_df_chan.itertuples():
                g.axes[channel-1, 0].axvspan(row.start_datetime, row.end_datetime, color='red', alpha=0.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True);
    return g

def plot_overlapped_fp(data, df_map=None, pal=None, shade_df=None, hspace=-0.5, height=3, aspect=12):
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    
    if pal is None:
        pal = sns.cubehelix_palette(16, rot=-.25, light=.7)
        
    xname = 'time' if 'time' in data.columns else 't'
    if xname not in data.columns:
        xname = 'datetime'
    yname = 'data' if 'data' in data.columns else 'd'
    
    assert xname in data.columns, f"xname {xname} not in data"
    assert yname in data.columns, f"yname {yname} not in data"

    if 'condition' in data.columns:
        g = sns.relplot(data=data, clip_on=False, x=xname, y=yname, hue='condition', 
                       palette=pal, row='channel', kind='line', linewidth=3, 
                       aspect=aspect, height=height)
    else:
        # Create base FacetGrid
        g = sns.FacetGrid(data=data, row='channel', aspect=aspect, height=height)
        # Add lineplot layer
        g.map(sns.lineplot, data=data, x=xname, y=yname, linewidth=3)
        # Add scatterplot layer if df_map is provided
        if df_map is not None:
            g.map(sns.scatterplot, data=df_map, x='datetime', y='plot_val', hue='channel')

    g.figure.subplots_adjust(hspace=hspace)

    if shade_df is not None:
        for channel in shade_df.channel.unique():
            shade_df_chan = shade_df.loc[shade_df.channel == channel]
            for row in shade_df_chan.itertuples():
                g.axes[channel-1, 0].axvspan(row.start_datetime, row.end_datetime, 
                                           color='red', alpha=0.3)
                
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    return g

def plot_basic_hypnogram(h, size=(20, 1), xlim=None, style_path=None, single_tone=False):
    
    state_colors = {}
    if single_tone:
        # GRAY SCALE
        state_colors['Wake'] = (1, "#333333")
        state_colors['NREM'] = (2, "#4f4f4f")
        state_colors['REM'] = (3, "#797979")
        
        # REDS
        state_colors['Wake'] = (1, "#6E2032")
        state_colors['NREM'] = (2, "#983F3F")
        state_colors['REM'] = (3, "#C88E87")
    
    else:
        state_colors['NREM'] = (2, "#4b71e3")
        state_colors['REM'] = (3, '#e34bde')
        state_colors['Wake'] = (1, '#4be350')
    
    plt.rcdefaults()
    if style_path is not None:
        plt.style.use(style_path)
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.bottom'] = False
    f, ax = plt.subplots(figsize=size)
    ax.set_ylim(0, 3)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(h.start_time.min(), h.end_time.max())
    
    # Add small epsilon to avoid exact overlap
    epsilon = pd.Timedelta(milliseconds=1)

    for i, bout in enumerate(h.itertuples()):
        if bout.state in state_colors.keys():
            value, color = state_colors[bout.state]
        else:
            value, color = state_colors['Wake']
        
        y_range = (value-1, value)
        y_range = (y_range[0]/3, y_range[1]/3)
        
        # Adjust end time of previous bout if there's an overlap
        start_time = bout.start_time
        end_time = bout.end_time
        
        # Ensure no overlap with previous bout
        if i > 0 and start_time <= h.iloc[i-1].end_time:
            # Set start time to just after previous bout's end time
            start_time = h.iloc[i-1].end_time + epsilon
        
        ax.axvspan(start_time, end_time, ymin=y_range[0], ymax=y_range[1], 
                color=color, alpha=1, linewidth=0)

    ax.set_yticks([])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return f, ax

def base_raster(data, xname='datetime', yname='negchan', pal=None, hspace=-0.5, color='blue', figsize=(24, 8)):
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    
    assert xname in data.columns, f"xname {xname} not in data"
    assert yname in data.columns, f"yname {yname} not in data"

    f, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(data, x=xname, y=yname, linewidth=0, alpha=0.7, s=60, ax=ax, color=color)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return f, ax
    
def base_trace_plot(data, xname='datetime', yname='data', color='blue', hspace=-0.5, height=3,  aspect=12):
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'     
        
    assert xname in data.columns, f"xname {xname} not in data"
    assert yname in data.columns, f"yname {yname} not in data"

    # Create line relplot
    g = sns.relplot(data, x=xname, y=yname, row='channel', linewidth=3, aspect=aspect, height=height, color=color, kind='line')

    g.figure.subplots_adjust(hspace=hspace)         
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    return g

def trace_plot(data, times, hspace=-0.6, figsize=(40, 12), color='blue'):
    """Quick plot of raw data traces

    Parameters
    ----------
    data : np.ndarray
        Raw data to plot, of shape (n_channels, n_samples)
    color : str, optional
        Color of the traces, by default SOM_BLUE
    hspace : float, optional
        Space between traces, by default -0.6
    figsize : tuple, optional
        Size of the figure, by default (28, 10)
    """
    f, ax = plt.subplots(data.shape[0], 1, figsize=figsize)
    for i in range(data.shape[0]):
        ax[i].plot(times, data[i, :], color=color)
    plt.subplots_adjust(hspace=hspace)
    return f, ax

def quick_trace_plot(data, times, stim_starts, stim_ends, color='blue', hspace=-0.6, figsize=(28, 10)):
    """Quick plot of raw data with stimulations

    Parameters
    ----------
    data : np.ndarray
        Raw data to plot, of shape (n_channels, n_samples)
    times : np.ndarray
        Times of the data, of shape (n_samples,)
    stim_starts : np.ndarray
        Start times of stimulations
    stim_ends : np.ndarray
        End times of stimulations
    color : str, optional
        Color of the traces, by default SOM_BLUE
    hspace : float, optional
        Space between traces, by default -0.6
    figsize : tuple, optional
        Size of the figure, by default (28, 10)
    """
    f, ax = plt.subplots(data.shape[0], 1, figsize=figsize)
    for i in range(data.shape[0]):
        ax[i].plot(times, data[i, :], color=color)
    plt.subplots_adjust(hspace=hspace)
    for on, off in zip(stim_starts, stim_ends):
        for a in ax:
            a.set_xlim(times[0], times[-1])
            a.axvspan(on, off, color='cornflowerblue', ymin=0.325, ymax=0.712, alpha=0.5)
    return f, ax

def mua_lfp_combo_plot(mua, lfp, hspace=-0.3, total_time=3, figsize=(40, 20), mua_color='blue', lfp_color='darkorange', mua_linewidth=2, lfp_linewidth=4, autolim=False, lfp_ylim=(-1500, 1500), mua_ylim=(-700, 700), chan_colors={}):
    """Quick plot of raw data traces

    Parameters
    ----------
    data : np.ndarray
        Raw data to plot, of shape (n_channels, n_samples)
    color : str, optional
        Color of the traces, by default SOM_BLUE
    hspace : float, optional
        Space between traces, by default -0.6
    figsize : tuple, optional
        Size of the figure, by default (28, 10)
    """
    total_time = 3
    t1 = np.linspace(0, total_time, mua.shape[1])
    t2 = np.linspace(0, total_time, lfp.shape[1])
    f, ax = plt.subplots(mua.shape[0], 1, figsize=figsize)
    
    for i in range(mua.shape[0]):
        if i+1 in chan_colors:
            mc = chan_colors[i+1]
            lc = chan_colors[i+1]
        else:
            mc = mua_color
            lc = lfp_color
        
        ax[i].plot(t1, mua[i, :], color=mc, linewidth=mua_linewidth)
        ax2 = ax[i].twinx()
        ax2.plot(t2, lfp[i, :], color=lc, linewidth=lfp_linewidth, alpha=0.8)
        
        ax[i].set_yticks([])
        ax2.set_yticks([])
        ax[i].set_yticklabels([])
        ax2.set_yticklabels([])
        
        if autolim==False:
            ax[i].set_ylim(mua_ylim)
            ax2.set_ylim(lfp_ylim)
        
    plt.subplots_adjust(hspace=hspace)
    return f, ax

def atomic_lfp(lfp_data, times=None, sems=None, hspace=-0.6, figsize=(24, 8), color='blue', line_alpha=1, linewidth=2):
    """Quick plot of raw LFP lfp_data traces

    Parameters
    ----------
    lfp_data : np.ndarray
        Raw lfp_data to plot, of shape (n_channels, n_samples)
    color : str, optional
        Color of the traces, by default blue
    hspace : float, optional
        Space between traces, by default -0.6
    figsize : tuple, optional
        Size of the figure, by default (24, 8)
    sems : np.ndarray, optional
        Semitransparent error bars, of shape (n_channels, n_samples)
    """
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['xtick.bottom'] = False
    
    if times is None:
        times = np.arange(lfp_data.shape[1])
    f, ax = plt.subplots(lfp_data.shape[0], 1, figsize=figsize)
    for i in range(lfp_data.shape[0]):
        ax[i].plot(times, lfp_data[i, :], color=color, alpha=line_alpha, linewidth=linewidth)
        ax[i].set_xlim(times[0], times[-1])
        if sems is not None:
            ax[i].fill_between(times, lfp_data[i, :]-sems[i, :], lfp_data[i, :]+sems[i, :], color=color, alpha=0.2)
    plt.subplots_adjust(hspace=hspace)
    return f, ax

def atomic_raster(mua_df, xname='datetime', yname='negchan', color='blue', alpha=0.7, s=60, figsize=(24, 8)):
    """Plot a raster plot of MUA data

    Parameters
    ----------
    mua_df : pl.DataFrame or pd.DataFrame
        the MUA data to plot, should have columns according to xname and yname
    xname : str, optional
        Column name for the x-axis, by default 'datetime'
    yname : str, optional
        Column name for the y-axis, by default 'negchan'
    color : str, optional
        Color of the raster plot, by default 'blue'
    figsize : tuple, optional
        Size of the figure, by default (24, 8)

    Returns
    -------
    f, ax : tuple
        Figure and axes objects
    """
    
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    
    
    
    assert xname in mua_df.columns, f"xname {xname} not in mua_df"
    assert yname in mua_df.columns, f"yname {yname} not in mua_df"

    
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(mua_df, x=xname, y=yname, linewidth=0, alpha=alpha, s=s, ax=ax, color=color)
    ax.set_xlim()
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return f, ax

def plot_lfp_mua_combined(lfp_data, mua_df, times, 
                          lfp_color='blue', mua_color='blue', 
                          mua_xname='datetime', mua_yname='negchan',
                          figsize=(36, 14), 
                          lfp_subplot_hspace=-0.4,
                          mua_height_multiplier=0.2,
                          lw=1.5, 
                          rsz=30,
                          lfp_alpha=1,
                          spike_alpha=0.7):
    """Plots LFP traces (each channel in its own subplot, stacked) 
    above an MUA raster plot, in a single figure.

    Parameters
    ----------
    lfp_data : np.ndarray
        Raw LFP data to plot, of shape (n_channels, n_samples).
    mua_df : pl.DataFrame or pd.DataFrame
        MUA data to plot. Needs columns specified by mua_xname and mua_yname.
    times : np.ndarray
        Time vector for the LFP data, same length as lfp_data.shape[1].
    lfp_color : str, optional
        Color of the LFP traces.
    mua_color : str, optional
        Color of the MUA raster plot.
    mua_xname : str, optional
        Column name in mua_df for MUA event times.
    mua_yname : str, optional
        Column name in mua_df for MUA event channels/depths.
    figsize : tuple, optional
        Size of the figure.
    lfp_subplot_hspace : float, optional
        Vertical spacing between all subplots (LFP channels and MUA plot). 
        Negative values (e.g., -0.5 or -0.6) can make LFP channel plots overlap, 
        similar to the original base_trace's hspace effect. Default is 0.0.

    Returns
    -------
    f : matplotlib.figure.Figure
        The created Matplotlib figure.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes. axs[0:-1] are LFP channel axes, axs[-1] is MUA axis.
    """
    
    num_lfp_channels = lfp_data.shape[0]
    if num_lfp_channels == 0:
        print("No LFP data to plot. Plotting MUA only.")
        f, mua_ax = plt.subplots(figsize=(figsize[0], figsize[1]/3.0 if figsize[1] and figsize[1] > 0 else 5))
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'None'
        sns.scatterplot(data=mua_df, x=mua_xname, y=mua_yname, linewidth=0, alpha=0.7, s=30, ax=mua_ax, color=mua_color)
        mua_ax.set_yticks([])
        mua_ax.set_yticklabels([])
        mua_ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=10)
        mua_ax.set_xlabel(str(mua_xname).capitalize(), fontsize=12)
        mua_ax.spines['top'].set_visible(False)
        mua_ax.spines['right'].set_visible(False)
        mua_ax.spines['left'].set_visible(False)
        plt.tight_layout()
        return f, np.array([mua_ax])

    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    #plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    plt.rcParams['ytick.left'] = False

    total_rows = num_lfp_channels + 1 
    f = plt.figure(figsize=figsize)
    
    mua_height_ratio = max(1, num_lfp_channels * mua_height_multiplier) 
    height_ratios = [1] * num_lfp_channels + [mua_height_ratio]
    
    gs = f.add_gridspec(total_rows, 1, height_ratios=height_ratios, hspace=lfp_subplot_hspace)
    
    axs = np.empty(total_rows, dtype=object)

    for i in range(num_lfp_channels):
        axs[i] = f.add_subplot(gs[i, 0], sharex=axs[0] if i > 0 else None)
        axs[i].plot(times, lfp_data[i, :], color=lfp_color, alpha=lfp_alpha, linewidth=lw)
        if i == 0: # Set xlim only for the first plot, others will share
            axs[i].set_xlim(times[0], times[-1])
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])
        axs[i].spines['bottom'].set_visible(False)
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    
    axs[-1] = f.add_subplot(gs[-1, 0], sharex=axs[0])
    mua_ax = axs[-1]
    sns.scatterplot(data=mua_df, x=mua_xname, y=mua_yname, linewidth=0, alpha=spike_alpha, s=rsz, ax=mua_ax, color=mua_color)
    mua_ax.set_yticks([])
    mua_ax.set_yticklabels([])
    mua_ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=14)
    mua_ax.set_xlabel(str(mua_xname).capitalize(), fontsize=12)
    mua_ax.spines['bottom'].set_visible(True)

    # Adjust the position of the last axes to move it further down
    pos = mua_ax.get_position()  # Get the current position
    mua_ax.set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height])
    
    return f, axs

def plot_lfp_counts_combined(lfp_data, mua_da, times, 
                          lfp_color='blue', mua_color='blue', 
                          mua_xname='datetime', mua_yname='negchan',
                          figsize=(36, 14), 
                          lfp_subplot_hspace=-0.4):
    """Plots LFP traces (each channel in its own subplot, stacked) 
    above an MUA raster plot, in a single figure.

    Parameters
    ----------
    lfp_data : np.ndarray
        Raw LFP data to plot, of shape (n_channels, n_samples).
    mua_df : pl.DataFrame or pd.DataFrame
        MUA data to plot. Needs columns specified by mua_xname and mua_yname.
    times : np.ndarray
        Time vector for the LFP data, same length as lfp_data.shape[1].
    lfp_color : str, optional
        Color of the LFP traces.
    mua_color : str, optional
        Color of the MUA raster plot.
    mua_xname : str, optional
        Column name in mua_df for MUA event times.
    mua_yname : str, optional
        Column name in mua_df for MUA event channels/depths.
    figsize : tuple, optional
        Size of the figure.
    lfp_subplot_hspace : float, optional
        Vertical spacing between all subplots (LFP channels and MUA plot). 
        Negative values (e.g., -0.5 or -0.6) can make LFP channel plots overlap, 
        similar to the original base_trace's hspace effect. Default is 0.0.

    Returns
    -------
    f : matplotlib.figure.Figure
        The created Matplotlib figure.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes. axs[0:-1] are LFP channel axes, axs[-1] is MUA axis.
    """
    
    num_lfp_channels = lfp_data.shape[0]
    if num_lfp_channels == 0:
        print("No LFP data to plot. Plotting MUA only.")
        f, mua_ax = plt.subplots(figsize=(figsize[0], figsize[1]/3.0 if figsize[1] and figsize[1] > 0 else 5))
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'None'
        sns.scatterplot(data=mua_df, x=mua_xname, y=mua_yname, linewidth=0, alpha=0.7, s=30, ax=mua_ax, color=mua_color)
        mua_ax.set_yticks([])
        mua_ax.set_yticklabels([])
        mua_ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=10)
        mua_ax.set_xlabel(str(mua_xname).capitalize(), fontsize=12)
        mua_ax.spines['top'].set_visible(False)
        mua_ax.spines['right'].set_visible(False)
        mua_ax.spines['left'].set_visible(False)
        plt.tight_layout()
        return f, np.array([mua_ax])

    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'None'
    plt.rcParams['ytick.left'] = False

    total_rows = num_lfp_channels + 1 
    f = plt.figure(figsize=figsize)
    
    mua_height_ratio = max(1, num_lfp_channels * 0.2) 
    height_ratios = [1] * num_lfp_channels + [mua_height_ratio]
    
    gs = f.add_gridspec(total_rows, 1, height_ratios=height_ratios, hspace=lfp_subplot_hspace)
    
    axs = np.empty(total_rows, dtype=object)

    for i in range(num_lfp_channels):
        axs[i] = f.add_subplot(gs[i, 0], sharex=axs[0] if i > 0 else None)
        axs[i].plot(times, lfp_data[i, :], color=lfp_color, linewidth=1.5)
        if i == 0: # Set xlim only for the first plot, others will share
            axs[i].set_xlim(times[0], times[-1])
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])
        axs[i].spines['bottom'].set_visible(False)
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    axs[-1] = f.add_subplot(gs[-1, 0], sharex=axs[0])
    mua_ax = axs[-1]

    
    mua_da.plot.imshow(ax=mua_ax, cmap='hot', origin='upper')
    if mua_ax.images:  # Check if there are any images on the axes#remove the scale bar
        last_image = mua_ax.images[-1]  # Get the last image artist
        if hasattr(last_image, 'colorbar') and last_image.colorbar:
            last_image.colorbar.remove()
    

    
    mua_ax.set_yticks([])
    mua_ax.set_yticklabels([])
    mua_ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=14)
    mua_ax.set_xlabel(str(mua_xname).capitalize(), fontsize=12)
    mua_ax.spines['bottom'].set_visible(True)

    # Adjust the position of the last axes to move it further down
    pos = mua_ax.get_position()  # Get the current position
    mua_ax.set_position([pos.x0, pos.y0 - 0.03, pos.width, pos.height])
    
    return f, axs

import matplotlib.colors as mcolors
import matplotlib
def shade_oodf(axs, oodf, off_color='cyan', on_color='green', alpha=0.2, single_ax=False, lw=3):
    if single_ax:
        axs = [axs]
    if 'channel' in oodf.columns:
        raise ValueError("oodf should not have channel column")
    for row in oodf.iter_rows(named=True):
        if row['status'] == 'on':
            print(row['start_datetime'], row['end_datetime'])
            [ax.axvspan(row['start_datetime'], row['end_datetime'], color=on_color, alpha=alpha) for ax in axs]
        elif row['status'] == 'off':
            fill_clr = mcolors.to_rgba(off_color, alpha)
            [ax.axvspan(row['start_datetime'], row['end_datetime'], facecolor=fill_clr, edgecolor='purple', linewidth=lw) for ax in axs]
    return axs

def shade_oodf_single_chan(ax, scdf, negchan=-1, span=0.5, alpha=0.7):
    for channel in scdf['channel'].unique():
        chdf = scdf.filter(pl.col('channel') == channel)
        chval = channel*negchan
        for bout in chdf.iter_rows(named=True):
            ax.fill_betweenx([chval - span, chval + span], bout['start_datetime'], bout['end_datetime'], color='orange' if bout['status'] == 'off' else 'green', lw=0.5, alpha=alpha)
    return ax

def _plot_histos_from_da(da):
    f, ax = plt.subplots(1, 1, figsize=(45, 15))
    max_bin = float(np.quantile(da.data, 0.95))
    min_bin = float(np.quantile(da.data, 0.40))
    bins = np.linspace(
        min_bin,
        max_bin,
        20,
    )

    h_chan = histogram(
                da,
                dim=["datetime"],
                bins=bins,
                density=False,
            )
    print(h_chan.shape)
    h_chan = h_chan / h_chan.max(dim=h_chan.dims[1])
    h_chan.plot(ax=ax)
    
    return f, ax

def custom_diverging_cmap(low='black', mid='#cdcdcd', high='royalblue', resolution=256):
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [low, mid, high],
        N=resolution                                  
    )
    return cmap