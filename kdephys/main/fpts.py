import xarray as xr
import numpy as np
from holoviews.plotting.links import RangeToolLink
import holoviews as hv
import panel as pn
import plotly.express as px


def x2df(xr, name=None, cols_2_drop=['time', 'timedelta', 'channel']):
    if name is not None:
        df = xr.to_dataframe(name='Power').drop(labels=cols_2_drop, axis=1)
    else: 
        df = xr.to_dataframe().drop(labels=cols_2_drop, axis=1) 
    return df


@xr.register_dataarray_accessor('fpts') 
class fpts_accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def ch(self, chan):
        return self._obj.sel(channel=chan)
    
    def from_start(self, mins):
        start = self._obj.datetime.values[0]
        td = np.timedelta64(mins, 'm')
        end = start+td
        dt_slice = slice(start, end)
        return self.obj_.sel(datetime=dt_slice)
    
    def from_start_sec(self, secs):
        start = self._obj.datetime.values[0]
        td = np.timedelta64(secs, 's')
        end = start+td
        dt_slice = slice(start, end)
        return self._obj.sel(datetime=dt_slice)
    
    def dts(self, t1, t2):
        date = str(self._obj.datetime.values.min())[0:10]
        if len(t1) == 6:
            t1 = t1[0:2]+':'+t1[2:4]+':'+t1[4:6]
            t2 = t1[0:2]+':'+t1[2:4]+':'+t1[4:6]
        elif len(t1) == 4:
            t1 = t1[0:2]+':'+t1[2:4]+':00'
            t2 = t2[0:2]+':'+t2[2:4]+':00'
        elif len(t1) == 2:
            t1 = t1[0:2]+':00:00'
            t2 = t2[0:2]+':00:00'
        
        datetime1 = date+'T'+t1
        datetime2 = date+'T'+t2
        dt_slice = slice(datetime1, datetime2)
        return self.sel(datetime=dt_slice)
    
    def qc_explore(self, chan=None, ds=False):
        if chan is not None:
            x = self._obj.sel(channel=chan)
        
        if ds != False:
            x=x.resample(datetime=ds).mean()
        
        pn.extension()
        
        crv = hv.Curve(x)
        tgt = x.hvplot(x='datetime', width=1800, height=600, kind='line')
        src = crv.opts(width=1800, height=200, yaxis=None)
        RangeToolLink(src, tgt)
        layout = (tgt + src).cols(1)
        layout.opts(shared_axes=False, merge_tools=False)
        return layout
    

    def xc(self, chan=5, ds=False):
        if chan is not None:
            x = self._obj.sel(channel=chan)
        
        if ds != False:
            x=x.resample(datetime=ds).mean()
        
        xdf = x2df(x, name='data')

        fig = px.line(xdf)
        fig.show()