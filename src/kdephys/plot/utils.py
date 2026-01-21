import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_lfp(data_array: xr.DataArray, title: str = 'Multi-Channel LFP Data', scale_factor: float = 1.0):
    """
    Plots a multi-channel line plot from an xarray.DataArray with dimensions 'channel' and 'time'.
    
    Parameters:
    - data_array: xarray.DataArray with dimensions 'channel' and 'time'
    - title: Title for the plot (default: 'Multi-Channel LFP Data')
    - scale_factor: Factor to scale the offset between channels (default: 1.0)
    
    Returns:
    - A matplotlib figure and axis with the plot.
    """
    plt.style.use('/home/kdriessen/gh_master/kdephys/kdephys/plot/dat_view.mplstyle')
    # Ensure the DataArray has the correct dimensions
    if 'channel' not in data_array.dims or 'time' not in data_array.dims:
        raise ValueError("DataArray must have 'channel' and 'time' dimensions.")
    
    n_channels = data_array.sizes['channel']
    times = data_array['time'].values
    lfp_data = data_array.values
    
    # Reverse the LFP data to reverse the order of channels
    lfp_data_reversed = lfp_data[::-1]
    
    # Calculate dynamic offsets based on the reversed data
    min_values = np.min(lfp_data_reversed, axis=1)
    max_values = np.max(lfp_data_reversed, axis=1)
    offsets = np.cumsum(scale_factor * np.abs(max_values[:-1] - min_values[1:]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(60, 40))

    for i in range(n_channels):
        if i == 0:
            ax.plot(times, lfp_data_reversed[i], label=f'Channel {n_channels-i}')
        else:
            ax.plot(times, lfp_data_reversed[i] + offsets[i-1], label=f'Channel {n_channels-i}')

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (with dynamic offset)')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Show the plot
    plt.show()
    
    return fig, ax

def plot_lfp_raw(data: np.array, title: str = 'Multi-Channel LFP Data', scale_factor: float = 1.0, fs=24414.0625, offset=0, ax=None):
    """
    Plots a multi-channel line plot from an xarray.DataArray with dimensions 'channel' and 'time'.
    
    Parameters:
    - data_array: xarray.DataArray with dimensions 'channel' and 'time'
    - title: Title for the plot (default: 'Multi-Channel LFP Data')
    - scale_factor: Factor to scale the offset between channels (default: 1.0)
    
    Returns:
    - A matplotlib figure and axis with the plot.
    """
    plt.style.use('/home/kdriessen/gh_master/kdephys/kdephys/plot/dat_view.mplstyle')
    # Ensure the DataArray has the correct dimensions
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    t1 = 0+offset
    t2 = (data.shape[1]/fs)+offset
    
    n_channels = data.shape[0]
    times = np.linspace(t1, t2, data.shape[1])
    lfp_data = data
    
    # Reverse the LFP data to reverse the order of channels
    lfp_data_reversed = lfp_data[::-1]
    
    # Calculate dynamic offsets based on the reversed data
    min_values = np.min(lfp_data_reversed, axis=1)
    max_values = np.max(lfp_data_reversed, axis=1)
    offsets = np.cumsum(scale_factor * np.abs(max_values[:-1] - min_values[1:]))
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(70, 40))

    
    for i in range(n_channels):
        if i == 0:
            ax.plot(times, lfp_data_reversed[i], label=f'Channel {n_channels-i}', linewidth=6)
        else:
            ax.plot(times, lfp_data_reversed[i] + offsets[i-1], label=f'Channel {n_channels-i}', linewidth=6)

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(' ')
    yts = len(ax.get_yticklabels())
    ylabs = np.linspace(data.shape[0]+1, 0, yts)
    ylabs = [int(i) for i in ylabs]
    ax.set_yticklabels(ylabs)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Show the plot
    #plt.show()
    
    return ax

def offset_raw_data(data: np.array, title: str = 'Multi-Channel LFP Data', scale_factor: float = 1.0, fs=24414.0625, offset=0):
    """
    Plots a multi-channel line plot from an xarray.DataArray with dimensions 'channel' and 'time'.
    
    Parameters:
    - data_array: xarray.DataArray with dimensions 'channel' and 'time'
    - title: Title for the plot (default: 'Multi-Channel LFP Data')
    - scale_factor: Factor to scale the offset between channels (default: 1.0)
    
    Returns:
    - A matplotlib figure and axis with the plot.
    """
    plt.style.use('/home/kdriessen/gh_master/kdephys/kdephys/plot/dat_view.mplstyle')
    # Ensure the DataArray has the correct dimensions
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    t1 = 0+offset
    t2 = (data.shape[1]/fs)+offset
    
    n_channels = data.shape[0]
    times = np.linspace(t1, t2, data.shape[1])
    lfp_data = data
    
    # Reverse the LFP data to reverse the order of channels
    lfp_data_reversed = lfp_data[::-1]
    
    # Calculate dynamic offsets based on the reversed data
    min_values = np.min(lfp_data_reversed, axis=1)
    max_values = np.max(lfp_data_reversed, axis=1)
    offsets = np.cumsum(scale_factor * np.abs(max_values[:-1] - min_values[1:]))
    
    offsets = np.insert(offsets, 0, 0)
    
    # add the offsets to the data
    for i in range(len(offsets)):
        lfp_data_reversed[i] = lfp_data_reversed[i] + offsets[i] 
    return lfp_data_reversed