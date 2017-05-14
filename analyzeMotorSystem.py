# Script to cluster motor subtypes
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import MotorContainer, SLHRepeatExperiment
from multiExpAnalysis import get_stack_types, dff, max_cluster, min_dist, create_centroid_stack
from typing import List
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
import matplotlib as mpl
from analyzeSensMotor import RegionResults


if __name__ == "__main__":
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    all_activity = np.array(dfile['all_activity'])
    # # rotate each line in all_activity
    # for i in range(all_activity.shape[0]):
    #     r = np.random.randint(30*5, 120*5)
    #     all_activity[i, :] = np.roll(all_activity[i, :], r)
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    exp_id = np.array(dfile['exp_id'])
    eid_nonan = exp_id[no_nan_aa]
    # limit sourceFiles to the contents of all_activity
    sourceFiles = [(g[0], e.original_time_per_frame) for e in exp_data for g in e.graph_info]
    sourceFiles = [sf for i, sf in enumerate(sourceFiles) if no_nan_aa[i]]
    tf_centroids = np.array(dfile['tf_centroids'])[no_nan_aa, :]
    dfile.close()
    # create motor containers
    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    itime = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    cat = exp_data[0].caTimeConstant
    mc_all = MotorContainer(sourceFiles, itime, cat, hdf5_store=tailstore)
    mc_flick = MotorContainer(sourceFiles, itime, cat, predicate=high_bias_bouts, tdd=mc_all.tdd)
    mc_fl_right = MotorContainer(sourceFiles, itime, cat, predicate=left_bias_bouts, tdd=mc_all.tdd)
    mc_fl_left = MotorContainer(sourceFiles, itime, cat, predicate=right_bias_bouts, tdd=mc_all.tdd)
    mc_swims = MotorContainer(sourceFiles, itime, cat, predicate=unbiased_bouts, tdd=mc_all.tdd)
    mc = [mc_all, mc_flick, mc_fl_right, mc_fl_left, mc_swims]
    labels = ["All motor", "Flicks", "Right flicks", "Left flicks", "Swims"]

    # perform stimulus-independent motor clustering and plot
    motor_corrs = np.zeros((all_activity.shape[0], 5))
    for i, act in enumerate(all_activity):
        for j, m in enumerate(mc):
            motor_corrs[i, j] = np.corrcoef(act, m[i])[0, 1]
    # find cells that have at least one above-threshold motor correlation
    motor_corrs[np.isnan(motor_corrs)] = 0
    # motor_corrs[motor_corrs < 0.6] = 0
    motor_cells = np.sum(motor_corrs > 0.6, 1) > 0
    motor_corrs_sig = motor_corrs[motor_cells, :].copy()
    # cluster by best regressor
    km = max_cluster(np.nanargmax(motor_corrs_sig, 1))

    # plot sorted by cluster identity
    fig, ax = pl.subplots()
    sns.heatmap(motor_corrs_sig[np.argsort(km.labels_), :], vmin=0, vmax=1, yticklabels=2500,
                xticklabels=labels, cmap="viridis")
    # plot cluster boundaries
    covered = 0
    for i in range(km.n_clusters):
        covered += np.sum(km.labels_ == i)
        ax.plot([0, motor_corrs_sig.shape[1] + 1], [km.labels_.size-covered, km.labels_.size-covered], 'k')
    ax.set_title('Above threshold correlations clustered and sorted')

    # slow POA ON regressor for finding heat-specific motor units at different timescales
    # NOTE: Initially tested with trigeminal ON activity as well but POA seems to recover more cells
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    poa_data = pickle.loads(np.array(storage["POA"]))
    poa_sens = np.mean(poa_data.region_acts[poa_data.region_mem == 1, :], 0)
    storage.close()
    # we create the following two kinds of motor regressors:
    # 1) A multiplication of mc_all with the activity trace (weighted by dFF)
    # 2) A multiplication of mc_all with a 0/1 thresholded version of the activity trace (only events when active)
    poa_thresh = 0.25  # POA slow ON cells are considered active above this threshold, based on looking at average
    heat_motor_corrs = np.zeros((all_activity.shape[0], 2))
    for i, act in enumerate(all_activity):
        # poa_act_mod_reg = mc_all[i] * poa_sens
        poa_act_th_reg = mc_all[i] * (poa_sens > poa_thresh)
        poa_nact_th_reg = mc_all[i] * (poa_sens < poa_thresh)
        heat_motor_corrs[i, 0] = np.corrcoef(act, poa_act_th_reg)[0, 1]
        heat_motor_corrs[i, 1] = np.corrcoef(act, poa_nact_th_reg)[0, 1]
    heat_motor_corrs[np.isnan(heat_motor_corrs)] = 0
    # heat_motor_corrs[heat_motor_corrs < 0.6] = 0
    all_m_corrs = np.hstack((motor_corrs, heat_motor_corrs))
    motor_cells = np.sum(all_m_corrs > 0.6, 1) > 0
    motor_corrs_sig = all_m_corrs[motor_cells, :].copy()

    km = max_cluster(np.nanargmax(motor_corrs_sig, 1))

    # plot sorted by cluster identity
    labels = ["All motor", "Flicks", "Right flicks", "Left flicks", "Swims", "Stim only", "No stim"]
    fig, ax = pl.subplots()
    sns.heatmap(motor_corrs_sig[np.argsort(km.labels_), :], vmin=0, vmax=1, yticklabels=2500,
                xticklabels=labels, cmap="viridis")
    covered = 0
    for i in range(km.n_clusters):
        covered += np.sum(km.labels_ == i)
        ax.plot([0, motor_corrs_sig.shape[1] + 1], [km.labels_.size-covered, km.labels_.size-covered], 'k')
    ax.set_title('Above threshold correlations clustered and sorted')

    # create a membership array for plotting
    membership_motor = np.full_like(membership, -1)
    msh_nn = np.full_like(mship_nonan, -1)
    msh_nn[motor_cells] = km.labels_
    membership_motor[no_nan_aa] = msh_nn

    membership_heat_motor = membership_motor.copy()
    membership_heat_motor[membership_heat_motor < 5] = -1

    # compute and plot distributions of nearest-neighbor distances for stim-only vs. no-stim motor types
    mhm = membership_heat_motor[no_nan_aa]
    stim_only = tf_centroids[mhm == 5, :]
    no_stim = tf_centroids[mhm == 6, :]
    d_so_so = min_dist(stim_only, stim_only, avgSmallest=2)
    d_so_ns = min_dist(stim_only, no_stim, avgSmallest=2)
    d_ns_ns = min_dist(no_stim, no_stim, avgSmallest=2)

    pl.figure()
    sns.kdeplot(d_so_so)
    sns.kdeplot(d_so_ns)
    sns.kdeplot(d_ns_ns)
    pl.xlabel("Distance $\mu m$")
    pl.ylabel("Density")
    sns.despine()

    # plot motor map
    stypes = get_stack_types(exp_data)
    stypes_nonan = stypes[no_nan_aa]
    motormem_nonan = membership_motor[no_nan_aa]
    fig, ax = pl.subplots()
    ax.scatter(tf_centroids[np.logical_and(stypes_nonan == "MAIN", motormem_nonan > -1), 0],
               tf_centroids[np.logical_and(stypes_nonan == "MAIN", motormem_nonan > -1), 1], s=2, alpha=0.1, c='k',
               label="All motor cells")
    ax.scatter(tf_centroids[np.logical_and(stypes_nonan == "MAIN", motormem_nonan == 5), 0],
               tf_centroids[np.logical_and(stypes_nonan == "MAIN", motormem_nonan == 5), 1], s=2, c="C0",
               label="No stim motor")
    ax.scatter(tf_centroids[np.logical_and(stypes_nonan == "MAIN", motormem_nonan == 6), 0],
               tf_centroids[np.logical_and(stypes_nonan == "MAIN", motormem_nonan == 6), 1], s=2, c="C1",
               label="Stim only motor")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)
