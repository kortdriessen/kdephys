import numpy as np
import pandas as pd
import tdt
import xarray as xr
import kd_analysis.signal.timefrequency as tfr
from scipy.ndimage import gaussian_filter
import kd_analysis.main.hypno as kh

bd = {}
bd['delta'] = slice(0.75, 4.1)
bd['theta'] = slice(4.1, 8.1)
bd['alpha'] = slice(8.1, 13.1)
bd['sigma'] = slice(11.1, 16.1)
bd['beta'] = slice(13.1, 30.1)
bd['gamma'] = slice(30.1, 100.1)

class ecdata(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(ecdata, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ecdata

    def ch(self, chan):
        """Return all bouts of the given states.

        Parameters:
        -----------
        states: list of str
        """
        return self.loc[self['channel'] == chan]

    def ts(self, slice_obj):
        """return dataframe values contained in slice_obj
        there should be a column called 'datetime' 
        """
        self = self.set_index('datetime')
        self = self.loc[slice_obj]
        return self.reset_index()

    def ri(self):
        return self.reset_index(drop=True)
    
    def bp_melt(self, bp_def=bd):
        """Melts a bandpower set to long-form.

        Parameters:
        -----------
        bp_def: bandpower dictionary
        """
        bp_melt = pd.melt(self, id_vars=['datetime', 'channel'], value_vars=list(bp_def.keys()))
        bp_melt.columns = ['datetime', 'channel', 'Band', 'Bandpower']
        return bp_melt
    
    def filt_state(self, states=['NREM']):
        """Filters a dataframe based on the state column

        Parameters:
        -----------
        bp_def: bandpower dictionary
        """
        return self[self.state.isin(states)].reset_index(drop=True)

    def sm(self, col):
        smoothed_data = gaussian_filter(self[col].values, 8)
        self[col+'_smooth'] = smoothed_data
        return self



def load_dataset(path_root, cond_list, type):
    ds = {}
    for cond in cond_list:
        path = path_root + cond + type + '.pkl'
        ds[cond] = ecdata(pd.read_pickle(path).reset_index())
    return ds

def load_hypnos(path_root, cond_list):
    h = {}
    for cond in cond_list:
        path = path_root + cond + '-hypno.pkl'
        h[cond] = pd.read_pickle(path)
    return h

def get_spike_df(path, t1, t2, chan, hyp, condition=''):
    def create_spike_xarray(tdt_obj):
        snip_times = tdt_obj.snips.eSpk.ts
        snip_times = snip_times.flatten()
        start_dt = pd.to_datetime(tdt_obj.info.start_date)
        full_dti = pd.to_timedelta(snip_times, 's') + start_dt
        phony_vals = np.ones(len(full_dti))
        xr_snips = xr.DataArray(phony_vals, coords=[full_dti], dims=['datetime'])
        return xr_snips

    def spk_xr_to_pd(xr_obj, condition):
        spkdf = pd.DataFrame()
        spkdf['datetime'] = xr_obj.datetime.values
        spkdf['spikes'] = xr_obj.values
        spkdf['state'] = xr_obj.state.values
        spkdf['condition'] = condition
        return spkdf
    
    tdt_obj = tdt.read_block(path, t1=t1, t2=t2, channel=chan, evtype=['snips'])
    xr_obj = create_spike_xarray(tdt_obj)
    xr_obj = kh.add_states(xr_obj, hyp)
    return ecdata(spk_xr_to_pd(xr_obj, condition=condition))

def tdt_to_pandas(path, t1=0, t2=0, channel=None, store=''):
    # Get the basic info needed from the TDT file:
    data = tdt.read_block(path, t1=t1, t2=t2, store=store, channel=channel)
    store = data.streams[store]
    info = data.info
    chan_cols = list(str(chan) for chan in channel)

    # Convert the TDT times to datetime objects:
    n_channels, n_samples = store.data.shape
    time = np.arange(0, n_samples) / store.fs
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta
    
    # Convert this data to a pandas dataframe. Each channel gets a column, datetime is the index:
    volts_to_microvolts = 1e6
    df = pd.DataFrame(store.data.T*volts_to_microvolts, columns=chan_cols)
    df['datetime'] = datetime
    df['timedelta'] = timedelta
    df['tdt_time'] = time
    #df = df.set_index('datetime')
    df.fs = store.fs
    return df

def pd_spg(df, window_length=4, overlap=2, **kwargs):
    
    # Get the raw data from the dataframe:
    d = df.drop(['tdt_time', 'timedelta', 'datetime'], axis=1, inplace=False)
    raw_data = d.to_numpy()
    fs = df.fs
    chans = list(d)
    chans_int = [int(i) for i in chans]
    
    # Compute the ecdata powergram:
    kwargs['nperseg'] = int(window_length * fs) # window length in number of samples
    kwargs['noverlap'] = int(overlap * fs) # overlap in number of samples
    kwargs['f_range'] = [0, 100] # frequency range to compute the spectrogram
    freqs, spg_time, spg = tfr.parallel_spectrogram_welch(
    raw_data, fs, **kwargs
    )
    tdt_time = df['tdt_time'].min() + spg_time
    timedelta = df['timedelta'].min() + pd.to_timedelta(spg_time, "s")
    datetime = df['datetime'].min() + pd.to_timedelta(spg_time, "s")
    xrda = xr.DataArray(
        spg,
        dims=("frequency", "datetime", "channel"),
        coords={
            "frequency": freqs,
            "datetime": datetime,
            "channel": chans_int,
            "timedelta": ("datetime", timedelta),
            "tdt_time": ("datetime", tdt_time),
        }
    )
    return ecdata(xrda.to_dataframe(name='spg').reset_index())


def pd_bp(spg_df, band_dict=bd):
    "expects a spectrogram dataframe from pd_spg"
    td_ix = np.repeat(pd.unique(spg_df['timedelta']), len(pd.unique(spg_df['channel'])))
    spg_df = spg_df.set_index(['frequency', 'datetime', 'channel'])

    #Create a new dataframe to hold the bandpower data:
    bp = spg_df.xs(slice(1,2), level='frequency', drop_level=False)
    bp = bp.groupby(level=['datetime', 'channel']).sum()
    bp_df = pd.DataFrame()
    
    #bp_df[['datetime', 'channel']] = spg_df[['datetime', 'channel']]

    # Calculate the power in each band:
    for band in band_dict:
        bp = spg_df.xs(band_dict[band], level='frequency', drop_level=False)
        bp = bp.groupby(level=['datetime', 'channel']).sum()
        bp_df[band] = bp['spg']
    
    bp_df['timedelta'] = td_ix
    return ecdata(bp_df.reset_index())

def combine_data_eeg(data, conds, dtype='bp'):
    for key in conds:
        data[key+'-e-'+dtype]['Condition'] = key
    data['concat'] = ecdata(pd.concat(list(data[key+'-e-'+dtype] for key in conds)))
    return data

def combine_data_lfp(data, conds, dtype='bp'):
    for key in conds:
        data[key+'-f-'+dtype]['Condition'] = key
    data['concat'] = ecdata(pd.concat(list(data[key+'-f-'+dtype] for key in conds)))
    return data

def add_states_to_data(data, hypno):
    dt = data.datetime.values
    states = hypno.get_states(dt)
    data['state'] = states
    return data

## INCOMPLETE -----------------------------------------------------------------------------------------------------------

def filter_data_by_state(data, state):
    return data[data.state == state]

def get_rel_bp_set(bp_set, hyp, times_cond):
    start = bp_set.datetime.values[0]
    t1 = times_cond['stim_on_dt']
    t2 = times_cond['stim_off_dt']
    avg_period = slice(start, t1)
    bp_bl = bp_set.ts(avg_period)