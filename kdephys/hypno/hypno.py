import pandas as pd
import yaml
from pathlib import Path
from kdephys.hypno.ecephys_hypnogram import Hypnogram, DatetimeHypnogram

def _infer_bout_start(df, bout):
    """Infer a bout's start time from the previous bout's end time.

    Parameters
    ----------
    h: DataFrame, (n_bouts, ?)
        Hypogram in Visbrain format with 'start_time'.
    row: Series
        A row from `h`, representing the bout that you want the start time of.

    Returns
    -------
    start_time: float
        The start time of the bout from `row`.
    """
    if bout.name == 0:
        start_time = 0.0
    else:
        start_time = df.loc[bout.name - 1].end_time

    return start_time


def load_hypno_file(path, st, dt=True):
    """Load a Visbrain formatted hypnogram."""
    df = pd.read_csv(path, sep="\t", names=["state", "end_time"], comment="*")
    df["start_time"] = df.apply(lambda row: _infer_bout_start(df, row), axis=1)
    df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
    if dt:
        df = to_datetime(df, st)
        return df
    else:
        return Hypnogram(df)


def to_datetime(df, start_datetime):
    df = df.copy()
    df["start_time"] = start_datetime + pd.to_timedelta(df["start_time"], "s")
    df["end_time"] = start_datetime + pd.to_timedelta(df["end_time"], "s")
    df["duration"] = pd.to_timedelta(df["duration"], "s")
    return DatetimeHypnogram(df)


def add_states(dat, hypnogram):
    """Annotate each timepoint in the dataset with the corresponding state label.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`.
    hypnogram: DatetimeHypnogram

    Returns:
    --------
    xarray object with new coordinate `state` on dimension `datetime`.
    """
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    
    if type(hypnogram) == pd.DataFrame:
        hypnogram = DatetimeHypnogram(hypnogram)
    
    states = hypnogram.get_states(dat.datetime)
    return dat.assign_coords(state=("datetime", states))


def get_states(hyp, times):
    """Given a hypnogram and an array of times, label each time with its state.
    Parameters:
    -----------
    times: (n_times,)
        The times to label.
    hyp: DatetimeHypnogram
    Returns:
    --------
    states (n_times,)
        The state label for each sample in `times`.
    """
    labels = pd.Series(["no_state"] * len(times))
    for bout in hyp.itertuples():
        times_in_bout = (times >= bout.start_time) & (times <= bout.end_time)
        labels.values[times_in_bout] = bout.state

    return labels


def no_states_array(times):
    """gives an array of no_state to match an array of times.
    Parameters:
    -----------
    times: (n_times,)
        The times to label.
    hyp: DatetimeHypnogram
    Returns:
    --------
    states (n_times,)
        The state label for each sample in `times`.
    """
    return pd.Series(["no_state"] * len(times))


def keep_states(dat, hypnogram, states):
    """Select only timepoints corresponding to desired states.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`
    hypnogram: DatetimeHypnogram
    states: list of strings
        The states to retain.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    try:
        assert "datetime" in dat.dims, "Data must contain datetime dimension."
    except:
        dat = dat.swap_dims({"time": "datetime"})
    keep = hypnogram.keep_states(states).covers_time(dat.datetime)
    return dat.sel(datetime=keep)


def keep_hypnogram_contents(dat, hypnogram):
    """Select only timepoints covered by the hypnogram.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`
    hypnogram: DatetimeHypnogram
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    keep = hypnogram.covers_time(dat.datetime)
    return dat.sel(datetime=keep)
