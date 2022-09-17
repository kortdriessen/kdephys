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

    def ch(self, chan):
        """Return only a selected channel.

        Parameters:
        -----------
        chan: list of channels to select
        """
        return self.loc[self["channel"] == chan]

    def ts(self, slice_obj):
        """
        --> return dataframe values with times contained in slice_obj

        --> needs a column called 'datetime'
        """
        self = self.set_index("datetime")
        self = self.loc[slice_obj]
        return self.reset_index()

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

    def filt_state(self, states=["NREM"]):
        """Filters a dataframe based on the state column

        Parameters:
        -----------
        bp_def: bandpower dictionary
        """
        return self[self.state.isin(states)].reset_index(drop=False)

    def sm(self, col, sigma=12):
        smoothed_data = gaussian_filter(self[col].values, sigma)
        self[col + "_smooth"] = smoothed_data
        return self
