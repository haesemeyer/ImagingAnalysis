# script to analyze and plot trial-to-trial correlations of neural activity and behavior
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib as mpl
import h5py
import pickle
from mh_2P import MotorContainer, SLHRepeatExperiment, trial_average, trial_to_trial_correlations
from typing import List
from typing import Dict
from analyzeSensMotor import RegionResults
import pandas
import os
from itertools import combinations


def shot_noise_model_act(act_mat, n_trials=3):
    t_avg = trial_average(act_mat, n_trials=n_trials)
    full_act = np.hstack([t_avg for i in range(n_trials)])
    return full_act + np.random.randn(full_act.shape[0], full_act.shape[1]) * np.sqrt(full_act)


def raw_trace_mat(experiments, units):
    """
    Get matrix of raw traces
    Args:
        experiments: n-cell long experiment indices
        units: n-cell long in-experiment cell indices

    Returns:
        n-cell x n-timepoints matrix of raw traces
    """
    raw = []
    for ID in np.unique(experiments):
        e = exp_data[ID]
        raw.append(e.get_raw_unit_traces(units[experiments == ID]))
    return np.vstack(raw)


def all_pairwise_correlations(data: np.ndarray, indices):
    """
    Computes and returns all pairwise correlations between the rows in data marked by indices
    """
    tests = set(combinations(indices, 2))
    return np.array([np.corrcoef(data[t[0], :], data[t[1], :])[0, 1] for t in tests])


def all_across_correlations(data: np.ndarray, ix_1: np.ndarray, ix_2: np.ndarray, skip_dict=None):
    """
    Returns the correlations of each row marked by ix_1 in data with each row in data markex by ix_2
    """
    all_corrs = []
    # Note: not vectorized to conserve memory
    for i in ix_1:
        for j in ix_2:
            if skip_dict is not None:
                if i in skip_dict and j in skip_dict[i]:
                    continue
                if j in skip_dict and i in skip_dict[j]:
                    continue
            all_corrs.append(np.corrcoef(data[i, :], data[j, :])[0, 1])
            if i in skip_dict:
                skip_dict[i].append(j)
            elif j in skip_dict:
                skip_dict[j].append(i)
            else:
                skip_dict[i] = [j]
    return np.array(all_corrs)


if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS1/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    all_activity = np.array(dfile['all_activity'])
    exp_id = np.array(dfile['exp_id'])
    eid = exp_id[no_nan_aa].astype(np.int)
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    # limit sourceFiles to the contents of all_activity
    sourceFiles = [(g[0], e.original_time_per_frame) for e in exp_data for g in e.graph_info]
    sourceFiles = [sf for i, sf in enumerate(sourceFiles) if no_nan_aa[i]]
    dfile.close()
    index_in_exp = np.hstack([np.arange(len(e.graph_info)) for e in exp_data])[no_nan_aa]

    # compare plane-to-plane behavioral variability within and across fish
    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    itime = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    # since raw starts are just 0s and 1s correlations aren't very meaningful. Therefore smooth with
    # 3.0s calcium kernel to approximate motor-cell responses
    motor_all_raw = MotorContainer(sourceFiles, itime, 3.0, hdf5_store=tailstore)
    in_exp_t_to_t = []
    cross_exp_t_to_t = []
    indices = np.arange(eid.size)
    plane_indices = []
    plane_eid = []
    done = {}
    for i in indices:
        if sourceFiles[i] in done:
            continue
        plane_indices.append(i)
        plane_eid.append(eid[i])
        done[sourceFiles[i]] = True
    plane_eid = np.array(plane_eid)
    all_data = np.vstack([motor_all_raw[i] for i in plane_indices])
    plane_indices = np.arange(all_data.shape[0])
    sdict = {}
    for eix in np.unique(plane_eid):
        ix_inexp = plane_indices[plane_eid == eix]
        ix_notexp = plane_indices[plane_eid != eix]
        in_exp_t_to_t.append(all_pairwise_correlations(all_data, ix_inexp))
        cross_exp_t_to_t.append(all_across_correlations(all_data, ix_inexp, ix_notexp, sdict))
        print(eix)
    in_exp_t_to_t = np.hstack(in_exp_t_to_t)
    cross_exp_t_to_t = np.hstack(cross_exp_t_to_t)
    fig, ax = pl.subplots()
    sns.kdeplot(in_exp_t_to_t, ax=ax, label="In-fish trial correlations")
    ax.plot(np.nanmedian(in_exp_t_to_t), 0.25, 'C0o')
    sns.kdeplot(cross_exp_t_to_t, ax=ax, label="Cross-fish trial correlations")
    ax.plot(np.nanmedian(cross_exp_t_to_t), 0.25, 'C1o')
    ax.set_xlabel("Behavior correlation")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)

    # analyze heat-cell trial-to-trial correlations by region
    test_labels = ["Trigeminal", "Rh6", "Rh2", "Cerebellum", "Habenula", "Pallium", "SubPallium", "POA"]
    region_results = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for k in test_labels:
        region_results[k] = (pickle.loads(np.array(storage[k])))
    storage.close()
    fig, ax = pl.subplots()
    for k in test_labels:
        c = trial_to_trial_correlations(region_results[k].region_acts[region_results[k].region_mem > -1, :], 3)
        sns.kdeplot(c, ax=ax, label=k)
    ax.legend()
    sns.despine(fig, ax)

    # compare the activity trial-to-trial correlations in each region to a shot-noise model
    noise_data = {}
    order = []
    all_shot = []
    all_heat = []
    for k in test_labels:
        if os.path.exists('H:/ClusterLocations_170327_clustByMaxCorr/raw_regions.hdf5'):
            raw_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/raw_regions.hdf5', 'r+')
        else:
            raw_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/raw_regions.hdf5', 'x')
        rr = region_results[k]
        ix_region = rr.region_indices[rr.region_mem > -1]
        if k in raw_file:
            rtm = np.array(raw_file[k])
        else:
            rtm = raw_trace_mat(eid[ix_region], index_in_exp[ix_region])
            raw_file.create_dataset(k, data=rtm)
            raw_file.flush()
        raw_file.close()
        sn_rtm = shot_noise_model_act(rtm)
        c1 = trial_to_trial_correlations(rtm, 3)
        noise_data[k] = c1
        order.append(k)
        all_heat.append(c1)
        c2 = trial_to_trial_correlations(sn_rtm, 3)
        all_shot.append(c2)
        noise_data[k+"_m"] = c2
        order.append(k+"_m")
    df = pandas.DataFrame(dict([(k, pandas.Series(v)) for k, v in noise_data.items()]))
    fig, ax = pl.subplots()
    sns.boxplot(data=df, ax=ax, order=order, orient='h')
    sns.despine(fig, ax)

    # compute and plot trial-to-trial correlations of non-clustered cells, heat-cells and motor cells brain wide
    all_shot = np.hstack(all_shot)
    all_heat = np.hstack(all_heat)
    heat_cells = np.logical_and(mship_nonan > -1, mship_nonan < 6)
    motor_cells = mship_nonan > 5
    unclass_cells = mship_nonan == -1
    c_motor_cells = trial_to_trial_correlations(all_activity[motor_cells, :], 3)
    c_unclass = trial_to_trial_correlations(all_activity[unclass_cells, :], 3)
    fig, ax = pl.subplots()
    sns.kdeplot(c_unclass, label="Other cells", ax=ax)
    ax.plot(np.median(c_unclass), 0.25, 'C0o')
    sns.kdeplot(all_heat, label="Heat modulated", ax=ax)
    ax.plot(np.median(all_heat), 0.25, 'C1o')
    sns.kdeplot(c_motor_cells, label="Motor cells", ax=ax)
    ax.plot(np.median(c_motor_cells), 0.25, 'C2o')
    sns.kdeplot(all_shot, label="SN model", ax=ax, color='k')
    ax.plot(np.median(all_shot), 0.25, 'ko')
    ax.set_xlabel("Activity correlation")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
