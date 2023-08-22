import yaml
import pandas as pd
import numpy as np


def check_sorting_thresholds(threshes=[4, 10, 3]):
    params_path = "/home/kdriessen/gh_master/pipeline_tdt/pipeline_tdt/params/params.yaml"
    with open(params_path, "r") as fp:
        params = list(yaml.safe_load_all(fp))
    proj_thresh = params[1]["analysis_params"]["ks2_5_base"]["projection_threshold"]
    detect_thresh = params[1]["analysis_params"]["ks2_5_base"]["detect_threshold"]
    proj_thresh.insert(0, detect_thresh)
    if threshes == proj_thresh:
        print("Thresholds match params.yaml, proceeding")
        return True
    else:
        print("Thresholds do not match params.yaml, updating params.yaml to", threshes)
        params[1]["analysis_params"]["ks2_5_base"]["projection_threshold"] = threshes[
            1:
        ]
        params[1]["analysis_params"]["ks2_5_base"]["detect_threshold"] = threshes[0]
        with open(params_path, "w") as fp:
            yaml.dump_all(params, fp)
        return True


def check_recs_and_times(subject, sort_id):
    ss = pd.read_excel(
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/spikesorting.xlsx",
        sheet_name="new",
    )
    ss_narrowed = ss.loc[np.logical_and(ss.subject == subject, ss.sort_id == sort_id)]
    if type(ss_narrowed.recording_end_times.values[0]) == int:
        times = [ss_narrowed.recording_end_times.values[0]]
        recs = [ss_narrowed.recordings.values[0]]
    else:
        times = ss_narrowed.recording_end_times.values[0].split(",")
        times = [int(t.strip()) for t in times]
        recs = ss_narrowed.recordings.values[0].split(",")
        recs = [r.strip() for r in recs]
    return times, recs
