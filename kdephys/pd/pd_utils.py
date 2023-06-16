from kdephys.pd.ecdata import ecdata
import pandas as pd
import xarray as xr
import numpy as np
import tdt
import kdephys.hypno as kh
import math


def load_saved_dataset(path_root, cond_list, type):
    ds = {}
    for cond in cond_list:
        path = path_root + cond + type + ".pkl"
        ds[cond] = ecdata(pd.read_pickle(path).reset_index())
    return ds


def load_saved_hypnos(path_root, cond_list):
    h = {}
    for cond in cond_list:
        path = path_root + cond + "-hypno.pkl"
        h[cond] = pd.read_pickle(path)
    return h


def tdt_to_pandas(path, t1=0, t2=0, channel=None, store=""):
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
    df = pd.DataFrame(store.data.T * volts_to_microvolts, columns=chan_cols)
    df["datetime"] = datetime
    df["timedelta"] = timedelta
    df["tdt_time"] = time
    # df = df.set_index('datetime')
    df.fs = store.fs
    return ecdata(df)


def combine_data_eeg(data, conds, dtype="bp"):
    for key in conds:
        data[key + "-e-" + dtype]["Condition"] = key
    data["concat"] = ecdata(pd.concat(list(data[key + "-e-" + dtype] for key in conds)))
    return data


def combine_data_lfp(data, conds, dtype="bp"):
    for key in conds:
        data[key + "-f-" + dtype]["Condition"] = key
    data["concat"] = ecdata(pd.concat(list(data[key + "-f-" + dtype] for key in conds)))
    return data


def add_states_to_data(data, hypno):
    ix_names = list(data.index.names)
    data = data.reset_index()
    dt = data.datetime.values
    states = hypno.get_states(dt)
    data["state"] = states
    return data.set_index(ix_names)


def add_states_to_dataset(dataset, hypnos):
    for key in dataset.keys():
        for cond in hypnos.keys():
            if cond in key:
                dataset[key] = add_states_to_data(dataset[key], hypnos[cond])
                break
    return dataset


## INCOMPLETE -----------------------------------------------------------------------------------------------------------


def filter_data_by_state(data, state):
    return data[data.state == state]


def get_rel_bp_set(bp_set, hyp, times_cond):
    start = bp_set.datetime.values[0]
    t1 = times_cond["stim_on_dt"]
    t2 = times_cond["stim_off_dt"]
    avg_period = slice(start, t1)
    bp_bl = bp_set.ts(avg_period)
