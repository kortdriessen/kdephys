import pandas as pd
import spikeinterface.extractors as se
import numpy as np
import pandas as pd
import pandas_flavor as pf

#----------------------------------------- for General Unit Dataframes ------------------------------------------

@pf.register_dataframe_method
def xnote(self, notes=["noisy", "bw"]):
    return self.loc[~self.note.str.contains("|".join(notes))]

@pf.register_dataframe_method
def rec(self, rec):
    return self.loc[self.recording == rec]

@pf.register_dataframe_method
def stm(self):
    return self.loc[self.stim == 1]

@pf.register_dataframe_method
def prb(self, probe='NNXr'):
    return self.loc[self.probe == probe]

@pf.register_dataframe_method
def pclus(self, probe='NNXr', cluster=0):
    return self.loc[np.logical_and(self.probe == probe, self.cluster_id == cluster)]

@pf.register_dataframe_method
def st(self, state):
    return self.loc[self.state == state]

@pf.register_dataframe_method
def sid(self, sort_id):
    return self.loc[self.sort_id == sort_id]

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
        if 'datetime' in self.columns:
            return self.loc[np.logical_and(self.datetime >= t1, self.datetime <= t2)]
        elif 'datetime' not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif type(t1) == np.datetime64 or type(t1) == pd.Timestamp:
        if 'datetime' in self.columns:
            return self.loc[np.logical_and(self.datetime >= t1, self.datetime <= t2)]
        elif 'datetime' not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif np.logical_or(type(t1) == int, type(t1) == float):
        if 'time' in self.columns:
            return self.loc[np.logical_and(self.time >= t1, self.time <= t2)]
        elif 'time' not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    else:
        print(f't1 and t2 must be strings, datetime64 or integers')

#----------------------------------------- METHODS FOR ON-OFF DATAFRAMES ------------------------------------------------------

@pf.register_dataframe_method
def oots(self, t1, t2):
    """slices the dataframe between t1 and t2, decides which column to slice based on the type of t1 and t2

    Args:
        t1 (str, datetime64, int): time to start slicing
        t2 (str, datetime64, int): time to end slicing
    """
    if type(t1) == str:
        t1 = np.datetime64(t1)
        t2 = np.datetime64(t2)
        if 'end_datetime' in self.columns:
            return self.loc[np.logical_and(self.end_datetime >= t1, self.end_datetime <= t2)]
        elif 'end_datetime' not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif type(t1) == np.datetime64 or type(t1) == pd.Timestamp:
        if 'end_datetime' in self.columns:
            return self.loc[np.logical_and(self.end_datetime >= t1, self.end_datetime <= t2)]
        elif 'end_datetime' not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    elif np.logical_or(type(t1) == int, type(t1) == float):
        if 'end_time' in self.columns:
            return self.loc[np.logical_and(self.end_time >= t1, self.end_time <= t2)]
        elif 'end_time' not in self.columns:
            return self.loc[np.logical_and(self.index >= t1, self.index <= t2)]
    else:
        print(f't1 and t2 must be strings, datetime64, integers, or floats')

@pf.register_dataframe_method
def off_only(self):
    return self.loc[self.state == 'off']

@pf.register_dataframe_method
def on_only(self):
    return self.loc[self.state == 'on']