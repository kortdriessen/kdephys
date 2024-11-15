import numpy as np
import pandas as pd
import datetime
import warnings
from pathlib import Path
from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
)

# NOTE - This is an exact copy of ecephys.hypnogram.hypnogram.py, done on 10/27/22 to remove dependency on ecephys


class Hypnogram:
    def __init__(self, df):
        self._df = df
        self._validate()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

    def _validate(self):
        if not {"state", "start_time", "end_time", "duration"}.issubset(self._df):
            raise AttributeError(
                "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
            )

    def keep_states(self, states):
        """Return all bouts of the given states.
        Parameters:
        -----------
        states: list of str
        """
        return self.__class__(self._df[self._df["state"].isin(states)])

    

    def hgts(self, start_time, end_time):
        """Keep all hypnogram bouts that fall between two times of day.
        Analagous to `pandas.DataFrame.between_time`.

        Paramters:
        ----------
        start_time:
            The starting hour, e.g. '13:00:00' for 1PM.
        end_time:
            The ending hour, e.g. '14:00:00' for 2PM.
        """ 
        return self.__class__(self.loc[(self['start_time']>=start_time)&(self['end_time']<=end_time)])
    
    def drop_states(self, states):
        """Drop all bouts of the given states.
        Parameters:
        -----------
        states: list of str
        """
        return self.__class__(self._df[~self._df["state"].isin(states)])

    def mask_times_by_state(self, times, states):
        """Return a mask that is true where times belong to specific states.
        Parameters
        ----------
        times: (n_times,)
            The times to mask.
        states: list of str
            The states of interest.
        Returns
        -------
        (n_times,)
            True where `times` belong to one of the indicated states, false otherise.
        """
        mask = np.full_like(times, False, dtype=bool)
        for bout in self.keep_states(states).itertuples():
            mask[(times >= bout.start_time) & (times <= bout.end_time)] = True

        return mask

    def get_states(self, times):
        """Given an array of times, label each time with its state.
        Parameters:
        -----------
        times: (n_times,)
            The times to label.
        Returns:
        --------
        states (n_times,)
            The state label for each sample in `times`.
        """
        labels = pd.Series([""] * len(times))
        for bout in self.itertuples():
            times_in_bout = (times >= bout.start_time) & (times <= bout.end_time)
            labels.values[times_in_bout] = bout.state

        return labels

    def covers_time(self, times):
        """Given an array of times, return True where that time is covered by
        the hypnogram."""
        covered = np.full_like(times, False, dtype="bool")
        for bout in self.itertuples():
            times_in_bout = (times >= bout.start_time) & (times <= bout.end_time)
            covered[times_in_bout] = True

        return covered

    def fractional_occupancy(self, ignore_gaps=True):
        """Return a DataFrame with the time spent in each state, as a fraction of
        the total time covered by the hypnogram.
        Parameters:
        -----------
        ignore_gaps: bool
            If True, unscored gaps do not contribute to total time.
        """
        total_time = (
            self.duration.sum()
            if ignore_gaps
            else self.end_time.max() - self.start_time.min()
        )
        return self.groupby("state").duration.sum() / total_time

    def write_htsv(self, file):
        """Write as HTSV."""
        file = Path(file)
        assert file.suffix == ".htsv", "File must use extension .htsv"
        file.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(
            file,
            sep="\t",
            header=True,
            index=False,
        )

    # TODO: Deprecated. Remove.
    def write(self, file):
        warnings.warn(
            "Hypnogram.write is deprecated and will be removed. Use Hypnogram.write_htsv instead."
        )
        self.write_htsv(file)

    def reconcile(self, other, how="self"):
        """Reconcile this hypnogram with another, per `reconcile_hypnograms`.

        Parameters:
        -----------
        other: Hypnogram
        how: str ('sel' or 'other')
            If 'self', resolve any conflicts in favor of this hypnogram.
            If 'other', resolve any conflicts in favor of `other`.
            Default: 'self'

        Returns:
        --------
        Hypnogram
        """
        assert type(self) == type(
            other
        ), "Cannot reconcile hypnograms of different types."
        if how == "self":
            return reconcile_hypnograms(self, other)
        elif how == "other":
            return reconcile_hypnograms(other, self)
        else:
            raise ValueError(
                f"Argument `how` should be either 'sel' or 'other'. Got {how}."
            )


class FloatHypnogram(Hypnogram):
    def write_visbrain(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(
            path, columns=["state", "end_time"], sep="\t", index=False, header=False
        )

    def as_datetime(self, start_datetime):
        df = self._df.copy()
        df["start_time"] = start_datetime + pd.to_timedelta(df["start_time"], "s")
        df["end_time"] = start_datetime + pd.to_timedelta(df["end_time"], "s")
        df["duration"] = pd.to_timedelta(df["duration"], "s")
        return DatetimeHypnogram(df)

    def keep_longer(self, duration):
        """Keep bouts longer than a given duration.

        Parameters:
        -----------
        duration: float
        """
        return self.__class__(self.loc[self.duration > duration])

    @classmethod
    def get_dummy(cls, start_time=0.0, end_time=np.Inf):
        """Return an empty, unscored hypnogram.

        Parameters
        ----------
        end_time: float
            The time at which the hypnogram should end, in seconds.

        Returns:
            H: pd.DataFrame
                A hypnogram containing a single state ("None") extending from t=0 until `end_time`.
        """
        df = pd.DataFrame(
            {
                "state": "None",
                "start_time": [float(start_time)],
                "end_time": [float(end_time)],
                "duration": [float(end_time - start_time)],
            }
        )
        return cls(df)

    @classmethod
    def from_htsv(cls, file):
        assert Path(file).suffix == ".htsv", "File must use extension .htsv"
        df = pd.read_csv(file, sep="\t", header=0)
        return cls(df)

    @classmethod
    def from_visbrain(cls, file):
        """Load a Visbrain formatted hypnogram."""
        df = pd.read_csv(file, sep="\t", names=["state", "end_time"], comment="*")
        df["start_time"] = df.apply(lambda row: _infer_bout_start(df, row), axis=1)
        df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
        return cls(df)

    @classmethod
    def from_Spike2(cls, file):
        """Load a Spike2 formatted hypnogram."""
        df = pd.read_table(
            file,
            sep="\t",
            names=["epoch", "start_time", "end_time", "state", "comment", "blank"],
            usecols=["epoch", "start_time", "end_time", "state"],
            index_col="epoch",
            skiprows=22,
        )
        return cls(df)

    @classmethod
    def from_SleepSign(cls, file):
        """Load a SleepSign hypnogram, exported using the `trend` function."""
        df = pd.read_table(
            file,
            skiprows=19,
            usecols=[0, 1, 2],
            names=["start_time", "epoch", "state"],
            parse_dates=["start_time"],
            index_col="epoch",
        )
        # Make sure that the data starts with epoch 0.
        assert (
            df.index.values[0] == 0
        ), "First epoch found is not #0. Unexpected number of header lines in file?"

        # The datetimes in the first column are meaningless. Convert them to floats.
        df["start_time"] = (df.start_time - df.start_time[0]) / pd.to_timedelta(1, "s")

        # Make sure all epochs are the same length, so that we can safely infer the file's end time.
        def _all_equal(iterator):
            """Check if all items in an un-nested array are equal."""
            try:
                iterator = iter(iterator)
                first = next(iterator)
                return all(first == rest for rest in iterator)
            except StopIteration:
                return True

        epoch_lengths = df.start_time.diff().values[1:]
        assert _all_equal(epoch_lengths), "Epochs are not all the same length."
        epoch_length = epoch_lengths[0]

        # Infer the epoch end times, and compute epoch durations
        df["end_time"] = df.start_time + epoch_length
        df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
        assert all(df.duration == epoch_length)

        # Reorder columns and return
        df = df[["state", "start_time", "end_time", "duration"]]
        return cls(df)
    
    def keep_first(self, cumulative_duration, trim=True):
        """Keep hypnogram bouts until a cumulative duration is reached.

        Parameters:
        -----------
        cumulative_duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        if trim:
            excess = self.duration.cumsum() - pd.to_timedelta(cumulative_duration).total_seconds()
            is_excess = excess > 0
            if not is_excess.any():
                return self
            amount_to_trim = excess[is_excess].min()
            trim_until = self.loc[is_excess].end_time.min() - amount_to_trim
            new = trim_hypnogram(self._df, self.start_time.min(), trim_until)
        else:
            keep = self.duration.cumsum() <= pd.to_timedelta(cumulative_duration).total_seconds()
            new = self.loc[keep]
        return self.__class__(new)
    
    def trim_select(self, start, end, ret_hyp=False):
        """Trim a hypnogram to start and end within a specified time range.
        Actually will truncate bouts if they extend beyond the range."""
        df = self._df.copy()
        if not {"state", "start_time", "end_time", "duration"}.issubset(df):
            raise AttributeError(
                "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
            )
        if not all(df["start_time"] <= df["end_time"]):
            raise ValueError("Not all start times precede end times.")
        if start > end:
            raise ValueError("Invalid value for kwargs: expected `start` <= `end`")

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
        return self.__class__(df)


class DatetimeHypnogram(Hypnogram):
    def as_float(self):
        df = self._df.copy()
        start_datetime = df.start_time.min()
        df["start_time"] = (df.start_time - start_datetime) / pd.to_timedelta("1s")
        df["end_time"] = (df.end_time - start_datetime) / pd.to_timedelta("1s")
        df["duration"] = df.duration / pd.to_timedelta("1s")
        return FloatHypnogram(df)

    def keep_first(self, cumulative_duration, trim=True):
        """Keep hypnogram bouts until a cumulative duration is reached.

        Parameters:
        -----------
        cumulative_duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        if trim:
            excess = self.duration.cumsum() - pd.to_timedelta(cumulative_duration)
            is_excess = excess > pd.to_timedelta(0)
            if not is_excess.any():
                return self
            amount_to_trim = excess[is_excess].min()
            trim_until = self.loc[is_excess].end_time.min() - amount_to_trim
            new = trim_hypnogram(self._df, self.start_time.min(), trim_until)
        else:
            keep = self.duration.cumsum() <= pd.to_timedelta(cumulative_duration)
            new = self.loc[keep]
        return self.__class__(new)

    def keep_last(self, cumulative_duration, trim=True):
        """Keep only a given amount of time at the end of a hypnogram.

        Parameters:
        -----------
        cumulative_duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        if trim:
            excess = self.duration[::-1].cumsum() - pd.to_timedelta(cumulative_duration)
            is_excess = excess > pd.to_timedelta(0)
            if not is_excess.any():
                return self
            amount_to_trim = excess[is_excess].min()
            trim_until = self.loc[is_excess].start_time.max() + amount_to_trim
            new = trim_hypnogram(self._df, trim_until, self.end_time.max())
        else:
            keep = np.cumsum(self.duration[::-1])[::-1] <= pd.to_timedelta(
                cumulative_duration
            )
            new = self.loc[keep]
        return self.__class__(new)
    
    
    def trim_select(self, start, end, ret_hyp=False):
        """Trim a hypnogram to start and end within a specified time range.
        Actually will truncate bouts if they extend beyond the range."""
        df = self._df.copy()
        if not {"state", "start_time", "end_time", "duration"}.issubset(df):
            raise AttributeError(
                "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
            )
        if not all(df["start_time"] <= df["end_time"]):
            raise ValueError("Not all start times precede end times.")
        if start > end:
            raise ValueError("Invalid value for kwargs: expected `start` <= `end`")

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
        return self.__class__(df)
    
    

    def _keep_between_time(self, start_time, end_time):
        """Keep all hypnogram bouts that fall between two times of day.
        Analagous to `pandas.DataFrame.between_time`.

        Paramters:
        ----------
        start_time:
            The starting hour, e.g. '13:00:00' for 1PM.
        end_time:
            The ending hour, e.g. '14:00:00' for 2PM.
        """
        start_time, end_time = _check_time(start_time), _check_time(end_time)
        keep = np.intersect1d(
            pd.DatetimeIndex(self.start_time).indexer_between_time(
                start_time, end_time
            ),
            pd.DatetimeIndex(self.end_time).indexer_between_time(start_time, end_time),
        )
        return self.__class__(self.iloc[keep])

    def keep_between_datetime(self, start_time, end_time):
        """Keep all hypnogram bouts that fall between two datetimes.

        Paramters:
        ----------
        start_time: datetime, or str
            The starting time, either as a datetime object or as a datetime string, e.g. '2021-12-30T22:00:01'
        end_time:
            The ending time, either as a datetime object or as a datetime string, e.g. '2021-12-30T22:00:01'
        """
        start_time, end_time = _check_datetime(start_time), _check_datetime(end_time)
        keep = (self.start_time >= start_time) & (self.end_time <= end_time)
        return self.__class__(self.loc[keep])

    def keep_longer(self, duration):
        """Keep bouts longer than a given duration.

        Parameters:
        -----------
        duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        return self.__class__(self.loc[self.duration > pd.to_timedelta(duration)])

    def get_consolidated(
        self,
        states,
        frac=0.8,
        minimum_time="0S",
        minimum_endpoint_bout_duration="0S",
        maximum_antistate_bout_duration=pd.Timedelta.max,
    ):
        """Get periods of consolidated sleep, wake, or any arbitrary set of states.

        A period is considered consolidated if more than a given fraction of its duration
        (e.g. frac=0.8 or 80%) is spent in the state(s) of interest, and the cumulative
        amount of time spent in the state(s) of interest exceeds `minimum_time`.
        Additionally, a consolidated period must be maximal, i.e. it cannot be contained by
        a longer consolidated period.

        Parameters:
        -----------
        states: list of str
            The states of interest.
        frac: float between 0 and 1
            The minimum fraction of a given period that must be spent in the states of
            interest for that period to be considered consolidated.
        minimum_time: timedelta format string
            The minimum cumulative time that must be spent in the states of interest for
            a given period to be considered consolidated.
        maximum_antistate_bout_duration: timedelta format string
            Do not allow periods to contain any bouts of unwanted states longer
            than a given duration.

        Returns:
        --------
        matches: list of pd.DataFrame
            Each DataFrame is a slice of the hypnogram, corresponding to a consolidated
            period.
        """
        # This method would be easy to adapt for FloatHypnogram types.
        assert (
            self.start_time.is_monotonic_increasing
        ), "Hypnogram must be sorted by start_time."
        minimum_time = pd.to_timedelta(minimum_time)
        maximum_antistate_bout_duration = pd.to_timedelta(
            maximum_antistate_bout_duration
        )
        endpoint_bouts = self.keep_states(states).keep_longer(
            minimum_endpoint_bout_duration
        )
        k = endpoint_bouts.index.min() - 1
        matches = list()
        # i = period start, j = period end, k = end of last consolidated period
        for i in endpoint_bouts.index:
            if i <= k:
                continue
            for j in endpoint_bouts.index[::-1]:
                if j < np.max([i, k]):
                    break
                isostate_bouts = self.__class__(self.loc[i:j]).keep_states(states)
                time_in_states = np.max(
                    [isostate_bouts.duration.sum(), pd.to_timedelta(0, "s")]
                )
                if time_in_states < minimum_time:
                    break  # because all other periods in the loop will also fail
                antistate_bouts = self.__class__(self.loc[i:j]).drop_states(states)
                if antistate_bouts.duration.max() > maximum_antistate_bout_duration:
                    continue
                total_time = (
                    self.loc[i:j].end_time.max() - self.loc[i:j].start_time.min()
                )
                if (time_in_states / total_time) >= frac:
                    matches.append(self.__class__(self.loc[i:j]))
                    k = j
                    break  # don't bother checking subperiods of good periods
        return matches

    def get_gaps(self, tolerance="0s"):
        """Get all unscored gaps in the hypnogram.

        Parameters:
        -----------
        tolterance: timedelta format string
            Optionally ignore gaps that are less than a given duration.

        Returns:
        --------
        gaps: list of dict
            Each gap detected, with start_time, end_time, and duration.
        """
        gaps = list()
        for i in range(len(self) - 1):
            current_bout_end = self.iloc[i].end_time
            next_bout_start = self.iloc[i + 1].start_time
            gap = next_bout_start - current_bout_end
            if gap > pd.to_timedelta(tolerance):
                gaps.append(
                    dict(
                        start_time=current_bout_end,
                        end_time=next_bout_start,
                        duration=gap,
                    )
                )

        return gaps

    def fill_gaps(self, tolerance="0s", fill_state="None"):
        """Fill all unscored gaps in the hypnogram with a specified state.

        Parameters:
        -----------
        tolerance: timedelta format string
            Optionally ignore gaps that are less than a given duration.
        fill_state: string
            The state to fill each gap with.

        Returns:
        --------
        hypnogram: DatetimeHypnogram
            The hypnogram, with gaps filled.
        """
        gaps = self.get_gaps(tolerance)
        for gap in gaps:
            gap.update({"state": fill_state})

        return self.__class__(
            pd.concat([self._df, pd.DataFrame.from_records(gaps)]).sort_values(
                "start_time", ignore_index=True
            )
        )
    @classmethod
    def from_htsv(cls, file):
        """Load a hypnogram whose entries are valid datetime strings."""
        assert Path(file).suffix == ".htsv", "File must use extension .htsv"
        try:
            df = pd.read_csv(file, sep="\t", header=0)
        except pd.errors.EmptyDataError:
            return None

        df["start_time"] = pd.to_datetime(df["start_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])
        df["duration"] = pd.to_timedelta(df["duration"])
        return cls(df)
    # ----------------------------------------------- KD-defined Methods of DatetimeHypnogram begin here -----------------------------------------------
    def states_by_duration(self, states, duration=None):
        self = self.keep_states(states)
        return self.keep_longer(duration) if duration else self
    
    def to_seconds(self):
        new = self.copy()
        new['start_time'] = (self.start_time - self.start_time.values[0]) / np.timedelta64(1, 's')
        new['end_time'] = (self.end_time - self.start_time.values[0]) / np.timedelta64(1, 's')
        return new
        
    


#####
# Misc. module functions
#####

def trim_hypnogram(df: pd.DataFrame, start, end, ret_hyp=False):
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
    if ret_hyp:
        return DatetimeHypnogram(df.reset_index(drop=True))
    else:
        return df.reset_index(drop=True)

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


def get_separated_wake_hypnogram(qwk_intervals, awk_intervals):
    """Turn a list of quiet wake and active wake intervals into a hypnogram.

    Parameters
    ----------
    qwk_intervals: list(tuple(float))
        Start and end times of each quiet wake bout.
    awk_intervals: list(tuple(float))
        Start and end times of each quiet wake bout.
    """
    qwk_intervals = np.asarray(qwk_intervals)
    awk_intervals = np.asarray(awk_intervals)

    qwk = pd.DataFrame(
        {
            "state": "qWk",
            "start_time": qwk_intervals[:, 0],
            "end_time": qwk_intervals[:, 1],
            "duration": np.diff(qwk_intervals).flatten(),
        }
    )
    awk = pd.DataFrame(
        {
            "state": "aWk",
            "start_time": awk_intervals[:, 0],
            "end_time": awk_intervals[:, 1],
            "duration": np.diff(awk_intervals).flatten(),
        }
    )

    df = pd.concat([qwk, awk]).sort_values(by=["start_time"]).reset_index()
    return Hypnogram(df)


def reconcile_hypnograms(h1, h2):
    """Combine two hypnograms such that any conflicts are resolved in favor of h1."""
    # Work with dataframes until the return, when we will cast as h1.__class__
    h1 = h1._df.copy()
    h2 = h2._df.copy()

    for index, row in h1.iterrows():
        # If h2 contains any interval exactly equivalent to this one, drop it.
        identical_intervals = (h2.start_time == row.start_time) & (
            h2.end_time == row.end_time
        )
        if any(identical_intervals):
            assert (
                sum(identical_intervals) == 1
            ), "More than one interval in h2 is identical to an interval found in 1. Is h2 well formed?"
            h2 = h2[~identical_intervals]

        # If h2 contains any intervals wholly contained by this one, drop them.
        sub_intervals = (h2.start_time >= row.start_time) & (
            h2.end_time <= row.end_time
        )
        if any(sub_intervals):
            h2 = h2[~sub_intervals]

        # If h2 contains any interval that whole contains this one, split it into preceeding (left) and succeeding (right) intervals.
        super_intervals = (h2.start_time <= row.start_time) & (
            h2.end_time >= row.end_time
        )
        if any(super_intervals):
            assert (
                sum(super_intervals) == 1
            ), "More than one interval in h2 wholly contains an interval found in h1. Is h2 well formed?"
            super_interval = h2[super_intervals]
            left_interval = super_interval.copy()
            left_interval.end_time = row.start_time
            left_interval.duration = left_interval.end_time - left_interval.start_time
            right_interval = super_interval.copy()
            right_interval.start_time = row.end_time
            right_interval.duration = (
                right_interval.end_time - right_interval.start_time
            )
            h2 = h2[~super_intervals]
            h2 = (
                h2.append([left_interval, right_interval])
                .sort_values("start_time")
                .reset_index(drop=True)
            )

        # If h2 contains any interval that overlaps the start of this interval, truncate it.
        left_intervals = (
            (h2.start_time < row.start_time)
            & (h2.end_time > row.start_time)
            & (h2.end_time < row.end_time)
        )
        if any(left_intervals):
            assert (
                sum(left_intervals) == 1
            ), "More than one interval in h2 overlaps the start of an interval found in h1. Is h2 well formed?"
            left_interval = h2[left_intervals]
            left_interval.end_time = row.start_time
            left_interval.duration = left_interval.end_time - left_interval.start_time
            h2[left_intervals] = left_interval

        # If h2 contains any interval that overlaps the endof this interval, adjust its start time.
        right_intervals = (
            (h2.start_time > row.start_time)
            & (h2.start_time < row.end_time)
            & (h2.end_time > row.end_time)
        )
        if any(right_intervals):
            assert (
                sum(right_intervals) == 1
            ), "More than one interval in h2 overlaps the end of an interval found in h1. Is h2 well formed?"
            right_interval = h2[right_intervals]
            right_interval.start_time = row.end_time
            right_interval.duration = (
                right_interval.end_time - right_interval.start_time
            )
            h2[right_intervals] = right_interval

    return h1.__class(h2.append(h1).sort_values("start_time").reset_index(drop=True))


def _check_datetime(dt):
    """Check that something is a valid datetime."""
    if is_datetime64_any_dtype(dt) or isinstance(dt, pd.Timestamp):
        return dt

    # Because pd.to_datetime will accept strings that don't contain a date, we have to check ourselves.
    if isinstance(dt, str):
        try:
            pd.core.tools.times.to_time(dt)
            warnings.warn(
                f"{dt} doesn't appear to include a date. Maybe you wanted `keep_between_time`?"
            )
        except ValueError:
            pass
        return pd.to_datetime(dt)

    raise ValueError("Unexpected datetime type.")


def _check_time(t):
    """Check that something is a valid time of day (e.g. 10:00:00, without a date)."""
    if isinstance(t, datetime.time):
        return t

    if isinstance(t, str):
        try:
            return pd.core.tools.times.to_time(t)
        except ValueError:
            raise ValueError(
                f"{t} could not be converted to a dateless time of day. Maybe you wanted `keep_between_datetime`?"
            )

    raise ValueError("Unexpected time of day type.")


#####
# Deprecated loading functions.
# TODO: Remove
#####


def load_visbrain_hypnogram(path):
    """Load a Visbrain formatted hypnogram."""
    warnings.warn(
        "load_visbrain_hypnogram is deprecated and will be removed. Use FloatHypnogram.from_visbrain instead."
    )
    return FloatHypnogram.from_visbrain(path)


def load_spike2_hypnogram(path):
    """Load a Spike2 formatted hypnogram."""
    warnings.warn(
        "load_spike2_hypnogram is deprecated and will be removed. Use FloatHypnogram.from_Spike2 instead."
    )
    return FloatHypnogram.from_Spike2(path)


def load_sleepsign_hypnogram(path):
    """Load a SleepSign hypnogram, exported using the `trend` function."""
    warnings.warn(
        "load_sleepsign_hypnogram is deprecated and will be removed. Use FloatHypnogram.from_SleepSign instead."
    )
    return FloatHypnogram.from_SleepSign(path)


def load_datetime_hypnogram(path):
    """Load a hypnogram whose entries are valid datetime strings."""
    warnings.warn(
        "load_datetime_hypnogram is deprecated and will be removed. Use DatetimeHypnogram.from_htsv instead."
    )
    return DatetimeHypnogram.from_htsv(path)
