import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import hypnogram as hp

from sklearn.decomposition import PCA
from neurodsp.plts.utils import check_ax
import sleep_score_for_me.utils.ssfm_utils as ssu

bp_def = dict(
    delta=(0.5, 4),
    theta=(4, 8),
    sigma=(11, 16),
    beta=(13, 30),
    low_gamma=(30, 55),
    high_gamma=(65, 90),
    omega=(300, 700),
)

"""
Functions Needed to do the actual scoring and build the hypnogram
-----------------------------------------------------------------
"""


def ssfm_v1(
    spg,
    emg_spg,
    nrem_percentiles=[50, 60, 65, 70],
    rem_percentiles=[60, 70, 80, 85],
    chan=2,
    f_range=slice(0, 100),
):
    """Uses the metrics and scoring techniques of Watson et al., 2016 (Neuron) to score a chunk of data into NREM, REM, and Wake

    spg --> xarray spectrogram containing the channel you wish to use for scoring (can contain other channels as well, as long as 'chan' option is used)
    emg_spg --> spectrogram of emg, channel 1 is always selected out by default
    chan --> the channel you want to use (should choose a parietal EEG if possible)
    f_range --> the range of frequencies to use for the PCA
    percentiles --> these are the percentiles you want to see plotted in order to determine the NREM and REM threshold values"""

    # First we get the EMG Band:
    emg_bands = kd.get_bp_set2(emg_spg, bands=bp_def, pandas=True)
    emg = emg_bands.omega

    # Then we get one channel in a given frequency range, and do the PCA
    spg_np = spg.sel(channel=chan, frequency=f_range).to_numpy().T
    pca = PCA(n_components=1)
    pca.fit(spg_np)
    pc = pca.transform(spg_np).flatten()

    # Then Theta Band:
    theta_narrow = (5, 10)
    theta_wide = (2, 16)
    thetaband = kd.get_bandpower(
        spg.sel(channel=chan), f_range=theta_narrow
    ) / kd.get_bandpower(spg.sel(channel=chan), f_range=theta_wide)
    thetaband = (thetaband / emg) / emg

    # Now we can construct the Dataframe which will be used to assign states
    dt_ix = spg.datetime.values
    scoring_df = pd.Series(pc, index=dt_ix)
    scoring_df = scoring_df.to_frame(name="PC1")
    scoring_df["Theta"] = thetaband.values
    scoring_df["EMG"] = emg.values
    scoring_df["state"] = np.nan

    # Now we need to figure out and set the thresholds for scoring out the NREM:
    hist, pcax = ssu.threshplot(
        pc, time=spg.datetime.values, percentiles=nrem_percentiles
    )

    nrem_threshold = float(input("Enter NREM Threshold: "))
    nrem_threshold = np.percentile(pc, nrem_threshold)

    # This is where we actually "score" the NREM based on a simple threshold value
    scoring_df.loc[scoring_df.PC1 >= nrem_threshold, "state"] = "NREM"

    # Now we need to set the threshold for REM::
    hist_rem, theta = ssu.threshplot(
        data=thetaband.values, time=spg.datetime.values, percentiles=rem_percentiles
    )

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
    m, d, g = ssu.plot_hypno_for_me(final_hypno, spg, emg_spg, bp_def)

    return hp.DatetimeHypnogram(final_hypno)
