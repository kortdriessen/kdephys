import pandas as pd
import yaml
from pathlib import Path
from kdephys.hypno.ecephys_hypnogram import Hypnogram, DatetimeHypnogram
import numpy as np

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

def get_states_fast(hyp, times, code=False):
    
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
    
    # Initialize with default values
    default_value = 0 if code else "no_state"
    labels = np.full(len(times), default_value, dtype=object if not code else int)

    # Convert times to numpy array if it's not already
    times_array = np.asarray(times)

    # Sort hypnogram by start_time for more efficient processing
    sorted_hyp = hyp.sort_values('start_time')

    # Use numpy searchsorted for efficient range checks
    start_times = sorted_hyp['start_time'].values
    end_times = sorted_hyp['end_time'].values
    states = sorted_hyp['state_code' if code else 'state'].values

    # Find indices where times fall within any bout
    start_indices = np.searchsorted(start_times, times_array, side='right') - 1
    end_indices = np.searchsorted(end_times, times_array, side='left')

    # Valid indices are where start_indices == end_indices
    valid_indices = (start_indices == end_indices) & (start_indices >= 0) & (start_indices < len(states))

    # Assign states to valid indices
    labels[valid_indices] = states[start_indices[valid_indices]]

    return pd.Series(labels, index=getattr(times, 'index', None))

def get_states(hyp, times, code=False):
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
    
    if code:
        labels = pd.Series([0] * len(times))
    else:
        labels = pd.Series(["no_state"] * len(times))
    for bout in hyp.itertuples():
        times_in_bout = (times >= bout.start_time) & (times <= bout.end_time)
        if code:
            labels.values[times_in_bout] = bout.state_code
        else:
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

def trim_hypnogram(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Trim a hypnogram to start and end within a specified time range.
    Actually will truncate bouts if they extend beyond the range."""
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")
    if start > end:
        raise ValueError("Invalid value for kwargs: expected `start` <= `end`")

    df = df.copy()
    starts_before = df["start_time"] < start
    df.loc[starts_before, "start_time"] = start
    ends_after = df["end_time"] > end
    df.loc[ends_after, "end_time"] = end
    starts_after = df["start_time"] >= df["end_time"]
    df = df[~starts_after]
    df["duration"] = df["end_time"] - df["start_time"]

    zero = np.array([0], dtype=df["duration"].dtype)[
        0
    ]  # Represents duration of length 0, regardless of dtype
    assert all(df["duration"] > zero)
    assert all(df["start_time"] >= start)
    assert all(df["end_time"] <= end)
    return df.reset_index(drop=True)

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
