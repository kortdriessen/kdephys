import polars as pl
import numpy as np
import pandas as pd
import math
import kdephys.unit_flavor as uf

def bout_duration_similarity_check(df, col="bout_duration"):
    """checks whether all values in the 'bout_duration' column are the same. If not, it checks for how different they are and if the difference is very small, it rectifies them so all values are the same.

    Parameters
    ----------
    df : pl.DataFrame
        polars dataframe with 'bout_duration' column
    """
    if all(
        df[col] == df[col][0]
    ):  # if all values are already equal, just return the dataframe
        return df
    else:
        mask = (
            df[col] == df[col][0]
        )  # if all vals not equal, check first how different they are; if the diff is small, rectify.
        for dur in df[
            col
        ]:  # this is a check that the difference between the bout_durations is very small.
            if dur != df[col][0]:
                diff = abs(dur - df[col][0])
                if diff < 1e-7:
                    continue
                else:
                    raise ValueError(
                        f"difference between bouts is too large! Diff = {diff}"
                    )
        new_val = df[col][
            0
        ]  # given that the difference between bouts is very small, we can just set all bouts to the first bout duration
        df = df.with_columns(
            pl.when(~mask).then(new_val).otherwise(pl.col(col)).alias(col)
        )
        assert all(df[col] == df[col][0])  # check that all values are now equal
        return df


@uf.register_pldf_method
def ch(self, chan: int):
    return self.filter(pl.col("channel") == chan)

@uf.register_pldf_method
def xch(self, chan: int):
    return self.filter(pl.col("channel") != chan)

@uf.register_pldf_method
def chnl(self, chan: int):
    return self.filter(pl.col("channel") == chan)

@uf.register_pldf_method
def sg(self, sg: str):
    return self.filter(pl.col("sub_group") == sg)

@uf.register_pldf_method
def st(self, state: str):
    return self.filter(pl.col("state") == state)


@uf.register_pldf_method
def cid(self, cid: int):
    return self.filter(pl.col("cluster_id") == cid)


@uf.register_pldf_method
def cid_un(self):
    return self["cluster_id"].unique()

@uf.register_pldf_method
def chunq(self):
    return self["channel"].unique()

@uf.register_pldf_method
def rec(self, rec):
    if type(rec) == str:
        return self.filter(pl.col("recording") == rec)
    elif type(rec) == list:
        return self.filter(pl.col("recording").is_in(rec))


@uf.register_pldf_method
def prb(self, probe_store: str):
    col = 'probe' if 'probe' in self.columns else 'store'
    return self.filter(pl.col(col) == probe_store)


@uf.register_pldf_method
def pclus(self, probe_store: str, cluster_id: int):
    return self.filter(
        (pl.col("probe") == probe_store) & (pl.col("cluster_id") == cluster_id)
    )

@uf.register_pldf_method
def pchan(self, probe_store: str, channel: int):
    return self.filter(
        (pl.col("probe") == probe_store) & (pl.col("channel") == channel)
    )


@uf.register_pldf_method
def prbs(self):
    return self["probe"].unique()


@uf.register_pldf_method
def bl_rec(self):
    """finds any recording with a -bl in the name

    Returns:
        str or list: baseline recording name(s)
    """
    recs = list(self["recording"].unique())
    bl_rec = [rec for rec in recs if "bl" in rec]
    if len(bl_rec) == 1:
        return bl_rec[0]
    elif len(bl_rec) > 1:
        return bl_rec
    else:
        print("No baseline recording found")


@uf.register_pldf_method
def ts(self, t1, t2):
    """slices the dataframe between t1 and t2, decides which column to slice based on the type f=of t1 and t2

    Args:
        t1 (str, datetime64, int): time to start slicing
        t2 (str, datetime64, int): time to end slicing
    """
    if type(t1) == str:
        t1 = np.datetime64(t1)
        t2 = np.datetime64(t2)
        assert "datetime" in self.columns
        return self.filter((pl.col("datetime") >= t1) & (pl.col("datetime") <= t2))
    elif type(t1) == np.datetime64 or type(t1) == pd.Timestamp:
        assert "datetime" in self.columns
        if type(t1) == pd.Timestamp:
            t1 = np.datetime64(str(t1))
            t2 = np.datetime64(str(t2))
        assert type(t1) == np.datetime64
        return self.filter((pl.col("datetime") >= t1) & (pl.col("datetime") <= t2))
    elif (type(t1) == int) or (type(t1) == float) or (type(t1) == np.float64):
        assert "time" in self.columns
        return self.filter((pl.col("time") >= t1) & (pl.col("time") <= t2))
    else:
        print(f"t1 and t2 must be strings, datetime64, floats, or integers")


@uf.register_pldf_method
def light_schedule(self):
    """returns a list with the light schedule for the recording

    Returns:
        times: lsit with light schedule
    """
    start = self["datetime"].to_pandas().min()
    end = self["datetime"].to_pandas().max()
    chunks = (end - start).total_seconds() / 3600 / 12
    chunks = math.ceil(chunks)  # round up to nearest integer
    begin = pd.Timestamp(f'{start.date().strftime("%Y-%m-%d")} 09:00:00')
    times = []
    for i in np.arange(chunks + 1):
        if i == 0:
            times.append(begin)
        else:
            time = times[-1] + pd.Timedelta("12h")
            times.append(time)
    return times


@uf.register_pldf_method
def frate(self, by="cluster_id"):
    """gets the firing rate over all bouts. Self should have columns 'probe', 'cluster_id', 'count', 'bout_duration'

    Parameters
    ----------
    by : str, optional
        get the firing rate by this parameter; DEFAULT = 'cluster_id'
    """
    assert "bout_duration" in self.columns
    assert "count" in self.columns
    if by == "cluster_id":
        new = self.groupby(["probe", "cluster_id"]).sum()
        return new.with_columns(
            (pl.col("count") / pl.col("bout_duration")).alias("fr")
        ).drop("count", "bout_duration")
    elif by == "probe":
        new = self.groupby(["probe", "cluster_id"]).sum()
        new = bout_duration_similarity_check(new)
        new = new.groupby(["probe", "bout_duration"]).sum()
        return new.with_columns(
            (pl.col("count") / pl.col("bout_duration")).alias("fr")
        ).drop("count", "bout_duration")


@uf.register_pldf_method
def frates(self):
    """simply turns count and bout_duration columns into a single 'fr' column, no grouping"""
    return self.with_columns(
        (pl.col("count") / pl.col("bout_duration")).alias("fr")
    ).drop("count", "bout_duration")


@uf.register_pldf_method
def oots(self, t1, t2):
    """slices the dataframe between t1 and t2, decides which column to slice based on the type f=of t1 and t2

    Args:
        t1 (str, datetime64, int): time to start slicing
        t2 (str, datetime64, int): time to end slicing
    """
    return self.filter(
        (pl.col("start_datetime") >= t1) & (pl.col("start_datetime") <= t2)
    )


@uf.register_pldf_method
def offs(self):
    return self.filter(pl.col("status") == "off")


@uf.register_pldf_method
def ons(self):
    return self.filter(pl.col("status") == "on")


@uf.register_pldf_method
def tz(self, tz):
    return self.filter(pl.col("time_zone") == tz)


@uf.register_pldf_method
def flt(self, col, val):
    return self.filter(pl.col(col) == val)

@uf.register_pldf_method
def xflt(self, col, val):
    return self.filter(pl.col(col) != val)

@uf.register_pldf_method
def cltz(self):
    """Clear the time_zone column"""
    return self.with_columns(pl.lit(None).alias("time_zone"))


@uf.register_pldf_method
def cl(self, col: str):
    """Clear any Column"""
    return self.with_columns(pl.lit(None).alias(col))

@uf.register_pldf_method
def cdn(self, cond: str):
    """filter on the condition column"""
    return self.filter(pl.col('condition')==cond)