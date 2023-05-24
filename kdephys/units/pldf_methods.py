import polars as pl
import numpy as np
import pandas as pd
import kdephys.unit_flavor as uf


@uf.register_pldf_method
def ch(self, chan: int):
    return self.filter(pl.col("channel") == chan)

@uf.register_pldf_method
def cid(self, cid: int):
    return self.filter(pl.col("cluster_id") == cid)

@uf.register_pldf_method
def cid_un(self):
    return self['cluster_id'].unique()

@uf.register_pldf_method
def rec(self, rec):
    if type(rec) == str:
        return self.filter(pl.col("recording") == rec)
    elif type(rec) == list:
        return self.filter(pl.col("recording").is_in(rec))

@uf.register_pldf_method
def prb(self, probe_store: str):
    return self.filter(pl.col("probe") == probe_store)

@uf.register_pldf_method
def pclus(self, probe_store: str, cluster_id: int):
    return self.filter((pl.col("probe") == probe_store) & (pl.col("cluster_id") == cluster_id))

@uf.register_pldf_method
def prbs(self):
    return self['probe'].unique()

@uf.register_pldf_method
def bl_rec(self):
    """finds any recording with a -bl in the name

    Returns:
        str or list: baseline recording name(s)
    """
    recs = list(self['recording'].unique())
    bl_rec = [rec for rec in recs if 'bl' in rec]
    if len(bl_rec) == 1:
        return bl_rec[0]
    elif len(bl_rec) > 1:
        return bl_rec
    else:
        print('No baseline recording found')

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
        assert 'datetime' in self.columns
        return self.filter((pl.col('datetime') >= t1) & (pl.col('datetime') <= t2))
    elif type(t1) == np.datetime64 or type(t1) == pd.Timestamp:
        assert 'datetime' in self.columns
        if type(t1) == pd.Timestamp:
            t1 = np.datetime64(str(t1))
            t2 = np.datetime64(str(t2))
        assert type(t1) == np.datetime64
        return self.filter((pl.col('datetime') >= t1) & (pl.col('datetime') <= t2))
    elif np.logical_or(type(t1) == int, type(t1) == float):
        assert 'time' in self.columns
        return self.filter((pl.col('time') >= t1) & (pl.col('time') <= t2))
    else:
        print(f't1 and t2 must be strings, datetime64, floats, or integers')