import pandas as pd
from scipy.ndimage import gaussian_filter
import kdephys.utils.spectral as sp

bands = sp.bands


class ecdata(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(ecdata, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ecdata

    def xc(self, chan):
        """Return only a selected channel.

        Parameters:
        -----------
        chan: list of channels to select
        """
        return self.xs(chan, level="channel", drop_level=False)

    def xt(self, time_tuple):
        """
        --> return dataframe values with times contained in time_tuple, which should contain datetime strings

        --> Index should have a 'datetime' level
        """
        t1 = time_tuple[0]
        t2 = time_tuple[1]
        time_slice = slice(t1, t2)
        return self.xs(time_slice, level="datetime", drop_level=False)

    def ri(self):
        return self.reset_index(drop=False)

    def bp_melt(self, bp_def=bands):
        """Melts a bandpower set to long-form.

        Parameters:
        -----------
        bp_def: bandpower dictionary, supplied automatically from kdephys.utils.spectral
        """
        bp_melt = pd.melt(
            self, id_vars=["datetime", "channel"], value_vars=list(bp_def.keys())
        )
        bp_melt.columns = ["datetime", "channel", "Band", "Bandpower"]
        return bp_melt

    def st(self, state="NREM"):
        """Filters a dataframe based on the state column

        Parameters:
        -----------
        bp_def: bandpower dictionary
        """
        if "state" in self.index.names:
            return self.xs(state, level="state", drop_level=False)
        else:
            print("state not in index names")

    def sm(self, col, sigma=12):
        smoothed_data = gaussian_filter(self[col].values, sigma)
        self[col + "_smooth"] = smoothed_data
        return self

    def bix(
        self, ix=["datetime", "channel", "state", "time", "timedelta", "condition"]
    ):
        """Set the index to a list of columns
        bix == 'Big Index'
        """
        return self.reset_index().set_index(ix)

    def avg(self, rel2, state="NREM", type="median"):
        if "state" not in self.index.names:
            self = self.reset_index().set_index(["datetime", "channel", "state"])
        if "state" not in rel2.index.names:
            rel2 = rel2.reset_index().set_index(["datetime", "channel", "state"])
        rel2_avg = (
            rel2.st(state).groupby(level="channel").median()
            if type == "median"
            else rel2.st(state).groupby(level="channel").mean()
        )
        return self / rel2_avg

    def dt(self):
        """return the datetime values of the index"""
        return self.reset_index().datetime.values
