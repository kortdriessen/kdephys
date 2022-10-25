import pandas as pd
import spikeinterface.extractors as se
import numpy as np
import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def xnote(self, notes=["noisy", "bw"]):
    return self.loc[~self.note.str.contains("|".join(notes))]


@pf.register_dataframe_method
def rec(self, rec):
    return self.loc[self.recording == rec]


@pf.register_dataframe_method
def stim(self):
    return self.loc[self.stim == 1]


@pf.register_dataframe_method
def st(self, state):
    return self.loc[self.state == state]


@pf.register_dataframe_method
def ts(self, t1, t2):
    return self.loc[np.logical_and(self.datetime > t1, self.datetime < t2)]
