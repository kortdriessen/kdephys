import pandas_flavor as pf
import kdephys.xr as kx
import pandas as pd
import math
import numpy as np

@pf.register_xarray_dataarray_method
def prb(self, store):
    """
    return dataset where store dimension (or coordinate) is prb
    """
    if 'store' in list(self.dims):
        return self.sel(store=store)
    elif 'store' in list(self.coords):
        return self.where(self.store==store, drop=True)
    else:
        print(f'there no dimension or coordinate named store in this dataset') 

@pf.register_xarray_dataarray_method
def ts(self, t1, t2):
    if 'datetime' in list(self.dims):
        return self.sel(datetime=slice(t1,t2))
    elif 'time' in list(self.dims):
        return self.sel(time=slice(t1,t2))
    else: 
        print(f'there no dimension named time or datetime in this dataarray')

@pf.register_xarray_dataarray_method
def rec(self, rec):
    return self.where(self.recording == rec, drop=True)

@pf.register_xarray_dataarray_method
def st(self, state):
    if type(state) == str:
        return self.where(self.state == state, drop=True)
    elif type(state) == list:
        return self.where(self.state.isin(state), drop=True)
    else:
        print('state must be a string or list of strings')

@pf.register_xarray_dataarray_method
def prn_dt(self):
    print(f'start -- {self.datetime.values.min()}')
    print(f'end -- {self.datetime.values.max()}')
    
@pf.register_xarray_dataarray_method
def smooth(self, sigma=14):
    return kx.utils.get_smoothed_da(self, sigma)

@pf.register_xarray_dataarray_method
def ch(self, chan=1):
    return self.sel(channel=chan)

@pf.register_xarray_dataarray_method
def light_schedule(self):
    start = pd.Timestamp(self.datetime.values.min())
    end = pd.Timestamp(self.datetime.values.max())
    
    chunks = (end - start).total_seconds() / 3600 / 12
    chunks = math.ceil(chunks) # round up to nearest integer
    begin = pd.Timestamp(f'{start.date().strftime("%Y-%m-%d")} 09:00:00' )
    times = []
    for i in np.arange(chunks+1):
        if i==0:
            times.append(begin)
        else:
            time = times[-1] + pd.Timedelta('12h')
            times.append(time)
    return times