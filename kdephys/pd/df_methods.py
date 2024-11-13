import pandas as pd
import spikeinterface.extractors as se
import numpy as np
import pandas as pd
import pandas_flavor as pf
from kdephys.pd.pd_utils import smooth

import kdephys.utils.spectral as sp
bands = sp.bands
# ----------------------------------------- for General Unit/Info Dataframes ------------------------------------------


@pf.register_dataframe_method
def xnote(self, notes=["noisy", "bw"]):
    return self.loc[~self.note.str.contains("|".join(notes))]


@pf.register_dataframe_method
def rec(self, rec):
    return self.loc[self.recording == rec]

@pf.register_dataframe_method
def sg(self, sg):
    return self.loc[self['sub_group'] == sg]

@pf.register_dataframe_method
def cnd(self, cond):
    return self.loc[self.condition == cond]

@pf.register_dataframe_method
def sbj(self, subject):
    return self.loc[self.subject == subject]

@pf.register_dataframe_method
def chnl(self, channel):
    return self.loc[self.channel == channel]

@pf.register_dataframe_method
def exp_als(self, exp):
    return self.loc[self.exp_alias == exp]


@pf.register_dataframe_method
def expmt(self, exp):
    return self.loc[self["exp"] == exp]

@pf.register_dataframe_method
def exptype(self, exptype):
    return self.loc[self["exp_type"] == exptype]

@pf.register_dataframe_method
def stm(self):
    return self.loc[self.stim == 1]


@pf.register_dataframe_method
def cid(self, cid):
    return self.loc[self["cluster_id"] == cid]


@pf.register_dataframe_method
def pclus(self, probe, cid):
    return self.loc[(self["cluster_id"] == cid) & (self["probe"] == probe)]


@pf.register_dataframe_method
def cid_un(self):
    return self["cluster_id"].unique()


@pf.register_dataframe_method
def prb(self, probe):
    if 'probe' in self.columns:
        return self.loc[self.probe == probe]
    elif 'store' in self.columns:
        return self.loc[self.store == probe]
    else:
        print('probe or store column not found')


@pf.register_dataframe_method
def prbs(self):
    return self["probe"].unique()


@pf.register_dataframe_method
def st(self, state):
    return self.loc[self.state == state]


@pf.register_dataframe_method
def sid(self, sort_id):
    return self.loc[self.sort_id == sort_id]

@pf.register_dataframe_method
def sm(self, col, sigma=12):
    new = self.copy()
    return smooth(new, col, sigma=sigma)

@pf.register_dataframe_method
def ts(self, t1, t2):
    """slices the dataframe between t1 and t2, decides which column to slice based on the type of t1 and t2

    Args:
        t1 (str, datetime64, int): time to start slicing
        t2 (str, datetime64, int): time to end slicing
    """
    if type(t1) == str:
        t1 = np.datetime64(t1)
        t2 = np.datetime64(t2)
        if "datetime" in self.columns:
            return self.loc[np.logical_and(self.datetime >= t1, self.datetime <= t2)]
        elif "datetime" not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif type(t1) == np.datetime64 or type(t1) == pd.Timestamp:
        if "datetime" in self.columns:
            return self.loc[np.logical_and(self.datetime >= t1, self.datetime <= t2)]
        elif "datetime" not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif np.logical_or(type(t1) == int, type(t1) == float):
        if "time" in self.columns:
            return self.loc[np.logical_and(self.time >= t1, self.time <= t2)]
        elif "time" not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    else:
        print(f"t1 and t2 must be strings, datetime64 or integers")

@pf.register_dataframe_method
def bpmlt(self, bp_def=bands):
    """Melts a bandpower set to long-form.

    Parameters:
    -----------
    bp_def: bandpower dictionary, supplied automatically from kdephys.utils.spectral
    """
    bp_melt = pd.melt(
        self, id_vars=["datetime", "channel"], value_vars=list(bp_def.keys())
    )
    bp_melt.columns = ["datetime", "channel", "band", "power"]
    return bp_melt

@pf.register_dataframe_method
def exp_names(self):
    """
    Gives all experiments a common name for plotting purposes. swi2 and swisin2 are replaced with swi and swisin
    """
    assert "exp" in self.columns, "exp column not found"
    new = self.copy()
    if 'swi2' in new.exp.unique():
        new.exp = new.exp.replace('swi2', 'swi')
    if 'swisin2' in new.exp.unique():
        new.exp = new.exp.replace('swisin2', 'swisin')
    return new


# ----------------------------------------- METHODS FOR ON-OFF DATAFRAMES ------------------------------------------------------


@pf.register_dataframe_method
def oots(self, t1, t2):
    """time selection for on-off dataframes

    Args:
        t1 (str, datetime64, int): time to start slicing
        t2 (str, datetime64, int): time to end slicing
    """
    if type(t1) == str:
        t1 = np.datetime64(t1)
        t2 = np.datetime64(t2)
        if "end_datetime" in self.columns:
            return self.loc[
                np.logical_and(self.end_datetime >= t1, self.end_datetime <= t2)
            ]
        elif "end_datetime" not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif type(t1) == np.datetime64 or type(t1) == pd.Timestamp:
        if "end_datetime" in self.columns:
            return self.loc[
                np.logical_and(self.end_datetime >= t1, self.end_datetime <= t2)
            ]
        elif "end_datetime" not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif np.logical_or(type(t1) == int, type(t1) == float):
        return self.loc[np.logical_and(self.start_time >= t1, self.end_time <= t2)]
    else:
        print(f"t1 and t2 must be strings, datetime64, integers, or floats")


@pf.register_dataframe_method
def hts(self, t1, t2):
    """time selection for hypnograms"""
    return self.loc[(self.start_time >= t1) & (self.end_time <= t2)]


@pf.register_dataframe_method
def offs(self):
    return self.loc[self.status == "off"]


@pf.register_dataframe_method
def ons(self):
    return self.loc[self.status == "on"]

@pf.register_dataframe_method
def mindur(self, min_dur):
    return self.loc[self.duration >= min_dur]

@pf.register_dataframe_method
def tz(self, time_zone):
    return self.loc[self.time_zone == time_zone]

@pf.register_dataframe_method
def dsc(self, descriptor):
    return self.loc[self['descriptor'] == descriptor]

@pf.register_dataframe_method
def cdn(self, cond):
    return self.loc[self['condition'] == cond]

@pf.register_dataframe_method
def bnd(self, band):
    return self.loc[self['Band'] == band] if 'Band' in self.columns else self.loc[self['band'] == band]