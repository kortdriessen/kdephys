import pandas as pd
import spikeinterface.extractors as se


def load_sorting_extractor(path, drop_noise=True):
    """load sorting extractor and info from kilosort output

    Args:
        path: path to ks output folder
        drop_noise (bool, optional): whether to drop noise clusters, loading only 'good' and 'mua'. Defaults to True.

    Returns:
        sorting: spikeinterface sorting extractor
        info: pandas dataframe with all cluster info
    """
    info_path = f"{path}/cluster_info.tsv"
    info = pd.read_csv(info_path, sep="\t")
    non_noise_info = info[info["group"] != "noise"]
    non_noise_ids = non_noise_info["cluster_id"].values

    sorting = se.KiloSortSortingExtractor(path)

    if drop_noise:
        sorting = sorting.select_units(non_noise_ids)
        return sorting, non_noise_info
    else:
        return sorting, info


def spikeinterface_sorting_to_dataframe(siSorting):
    """converts spikeinterface sorting extractor to dataframe of spike times with their cluster_id

    Args:
        siSorting: spikeinterface sorting extractor object

    Returns:
        spikes: pandas dataframe with columns 'time' (spike times) and cluster_id
    """
    [(spikeSamples, clusterIDs)] = siSorting.get_all_spike_trains()
    spikeTimes = spikeSamples / siSorting.get_sampling_frequency()
    spikes = pd.DataFrame(
        {
            "time": spikeTimes,
            "cluster_id": clusterIDs,
        }
    )
    return spikes
