import pandas as pd
import numpy as np
import tdt
import kdephys.hypno as kh
import xarray as xr
import kdephys.pd as kpd


def get_tdt_spikes(path, t1, t2, chan, hyp, condition=""):
    def create_spike_xarray(tdt_obj):
        snip_times = tdt_obj.snips.eSpk.ts
        snip_times = snip_times.flatten()
        start_dt = pd.to_datetime(tdt_obj.info.start_date)
        full_dti = pd.to_timedelta(snip_times, "s") + start_dt
        phony_vals = np.ones(len(full_dti))
        xr_snips = xr.DataArray(phony_vals, coords=[full_dti], dims=["datetime"])
        return xr_snips

    def spk_xr_to_pd(xr_obj, condition):
        spkdf = pd.DataFrame()
        spkdf["datetime"] = xr_obj.datetime.values
        spkdf["spikes"] = xr_obj.values
        spkdf["state"] = xr_obj.state.values
        spkdf["condition"] = condition
        return spkdf

    tdt_obj = tdt.read_block(path, t1=t1, t2=t2, channel=chan, evtype=["snips"])
    xr_obj = create_spike_xarray(tdt_obj)
    xr_obj = kh.add_states(xr_obj, hyp)
    return kpd.ecdata(spk_xr_to_pd(xr_obj, condition=condition))
