from typing import Type
import numpy as np
import pandas as pd

import scipy


"""
Functions Needed to do the actual scoring and build the hypnogram
-----------------------------------------------------------------
"""


def ssfm_v2(
    spg,
    emg_spg,
    nrem_percentiles=[50, 60, 65, 70],
    rem_percentiles=[60, 70, 80, 85],
    chan=2,
    ss=8,
    user_hyp=None,
    art=False,
):
    """
    Scores data into Wake, REM, NREM, and Transition-to-REM
    NOTE: Both spectrograms can have one channel or multiple.
    SSFM will always select EMG channel-1 if multiple are given, whereas the 'chan' aregument selects the data channel if multiple are given.

    spg --> xarray spectrogram containing the channel you wish to use for scoring (can contain other channels as well, as long as 'chan' option is used)
    emg_spg --> spectrogram of emg, channel 1 is always selected out by default
    chan --> the channel you want to use (should choose a parietal EEG if possible)
    f_range --> the range of frequencies to use for the PCA
    percentiles --> these are the percentiles you want to see plotted in order to determine the NREM and REM threshold values"""

    try:
        spg = spg.sel(channel=chan)
    except KeyError:
        print("Passing Channel Selection Error, Data Already Contains Single Channel")

    if art == True:
        art_hypno = artifactus_identicus(spg, emg_spg, bp_def, chan=chan, ss=ss)
        spg = artifactus_rejectus(spg, art_hypno)

    # This gets our datetime index evenly spaced so the hypnogram comes out concatenated correctly
    dt_original = spg.datetime.values
    start = dt_original.min()
    dt_freq = scipy.stats.mode(np.diff(dt_original)).mode[0]
    dt_freq = dt_freq / pd.to_timedelta(1, "ns")
    dt_freq = str(dt_freq) + "ns"
    new_dti = pd.date_range(start, periods=len(dt_original), freq=dt_freq)
    spg = spg.assign_coords(datetime=new_dti)

    # First we get the EMG Band:
    emg_bands = kd.get_bp_set2(emg_spg, bands=bp_def, pandas=True)
    try:
        emg = emg_bands.xs(1, level="cahnnel").omega
    except TypeError:
        emg = emg_bands.omega
    # emg = kd.get_smoothed_da(emg, smoothing_sigma=ss)

    # Then we get the delta metric to be used for NREM scoring:
    delta = kd.get_bandpower(spg, f_range=(0.5, 4))
    delta_smooth = kd.get_smoothed_da(delta, smoothing_sigma=ss)

    # Then Theta Band:
    theta_narrow = (5, 10)
    theta_wide = (2, 16)
    tn = kd.get_bandpower(spg, f_range=theta_narrow)
    tn = kd.get_smoothed_da(tn, smoothing_sigma=ss)
    tw = kd.get_bandpower(spg, f_range=theta_wide)
    tw = kd.get_smoothed_da(tw, smoothing_sigma=ss)
    thetaband = tn / tw
    thetaband = (thetaband / emg) / emg

    # Now we can construct the Dataframe which will be used to assign states
    dt_ix = spg.datetime.values
    scoring_df = pd.Series(delta_smooth, index=dt_ix)
    scoring_df = scoring_df.to_frame(name="Delta")
    scoring_df["Theta"] = thetaband.values
    scoring_df["EMG"] = emg.values
    scoring_df["state"] = np.nan

    # Now we need to figure out and set the thresholds for scoring out the NREM:
    hist, delta_ax = ssu.threshplot(
        delta_smooth, time=spg.datetime.values, percentiles=nrem_percentiles
    )
    print("Violet --> " + str(nrem_percentiles[0]))
    print("Blue --> " + str(nrem_percentiles[1]))
    print("Green --> " + str(nrem_percentiles[2]))
    print("Red --> " + str(nrem_percentiles[3]))

    nrem_threshold = float(input("Enter NREM Threshold: "))
    nrem_threshold = np.percentile(delta_smooth, nrem_threshold)

    # This is where we actually "score" the NREM based on a simple threshold value
    scoring_df.loc[scoring_df.Delta >= nrem_threshold, "state"] = "NREM"

    # Now we need to set the threshold for REM::
    hist_rem, theta = ssu.threshplot(
        data=thetaband.values,
        time=spg.datetime.values,
        percentiles=rem_percentiles,
        cut=95,
    )
    print("Violet --> " + str(rem_percentiles[0]))
    print("Blue --> " + str(rem_percentiles[1]))
    print("Green --> " + str(rem_percentiles[2]))
    print("Red --> " + str(rem_percentiles[3]))

    rem_threshold = float(input("Enter REM Threshold: "))
    rem_threshold = np.percentile(thetaband.values, rem_threshold)

    # This is where we actually "score" the REM based on a simple threshold value
    scoring_df.loc[
        np.logical_and(scoring_df.Theta >= rem_threshold, scoring_df.state != "NREM"),
        "state",
    ] = "REM"

    # Now we just score the rest of the hypnogram as Wake:
    scoring_df.loc[
        np.logical_and(scoring_df.state != "NREM", scoring_df.state != "REM"), "state"
    ] = "Wake"

    "At this point, all of the data is actually scored, and we can simply call build_hypno_for_me to get the start and end times and build the hypnogram"

    final_hypno = ssu.build_hypno_for_me(scoring_df["state"])

    # This eliminates any Wake --> REM direct transitions
    fhc = np.arange(0, len(final_hypno.state) - 1)
    for i in fhc:
        if (
            final_hypno.iloc[i].state == "Wake"
            and final_hypno.iloc[i + 1].state == "REM"
        ):
            final_hypno.iloc[
                i, final_hypno.columns.get_loc("state")
            ] = "Transition-to-REM"

    if user_hyp is not None:
        ssfm_ax, user_ax = ssu.compare_hypnos_for_me(spg, final_hypno, user_hyp)

    m, d, g = ssu.plot_hypno_for_me(final_hypno, spg, emg_spg, bp_def, chan=chan)
    return hp.DatetimeHypnogram(final_hypno)
