# Script to cluster motor subtypes
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import MotorContainer, SLHRepeatExperiment
from multiExpAnalysis import get_stack_types, min_dist
from typing import List
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
import matplotlib as mpl
from analyzeSensMotor import RegionResults


class MotorCluster:
    """
    Class to cluster our motor-data by maximal correlation but before assigning to a sub-category testing whether
    it is significantly better correlated to the sub-category than a broader category
    Flick_Left(Right) -> Flicks -> All motor
    Swims -> All motor
    Heat_only -> All motor
    NoHeat -> All motor
    Assumes the following column structure of correlations:
    All - Flicks - Right flicks - Left flicks - Swims - StimOnly - NoStim
    """
    def __init__(self, mc_corrs: np.ndarray, mc, all_activity, poa_on, poa_off, corr_thresh=0.6, p_cut=0.01):
        self.cluster_labels = np.full(mc_corrs.shape[0], -1)
        for i, c in enumerate(mc_corrs):
            if np.nanmax(c) < corr_thresh:
                continue
            primary = np.nanargmax(c)
            if primary == 0:
                self.cluster_labels[i] = 0
            else:
                # need to perform significance tests
                if primary == 1 or primary == 4:
                    # potential flick or swim
                    p = correlation_bootstrap_p(all_activity[i, :], mc[0][i], mc[primary][i])
                    if p < p_cut:
                        self.cluster_labels[i] = primary
                    else:
                        self.cluster_labels[i] = 0
                elif primary == 5:
                    # potential stim only
                    p = correlation_bootstrap_p(all_activity[i, :], mc[0][i], mc[0][i] * poa_on)
                    if p < p_cut:
                        self.cluster_labels[i] = primary
                    else:
                        self.cluster_labels[i] = 0
                elif primary == 6:
                    # potential no stim
                    p = correlation_bootstrap_p(all_activity[i, :], mc[0][i], mc[0][i] * poa_off)
                    if p < p_cut:
                        self.cluster_labels[i] = primary
                    else:
                        self.cluster_labels[i] = 0
                elif primary == 2 or primary == 3:
                    # potential flick right or left
                    p = correlation_bootstrap_p(all_activity[i, :], mc[0][i], mc[primary][i])
                    if p >= p_cut:
                        self.cluster_labels[i] = 0
                    else:
                        p = correlation_bootstrap_p(all_activity[i, :], mc[1][i], mc[primary][i])
                        if p < p_cut:
                            self.cluster_labels[i] = primary
                        else:
                            self.cluster_labels[i] = 1
                else:
                    raise ValueError("Non recognized cluster number")


def correlation_bootstrap_p(act, reg_1, reg_2, nboot=1000):
    """
    Computes the p-value that the correlation of act to reg_2 is larger than the correlation of act to reg_1
    Args:
        act: An activity trace
        reg_1: The first regressor
        reg_2: The second regressor
        nboot: The number of bootstrap samples to generate

    Returns:
        p_value that c(act, reg2) > c(act, reg1)
    """
    if act.size != reg_1.size or reg_1.size != reg_2.size:
        raise ValueError("All traces need to have the same length")
    ix = np.arange(act.size)
    all_d_corrs = np.zeros(act.size)
    for i in range(nboot):
        to_take = np.random.choice(ix, ix.size)
        all_d_corrs[i] = np.corrcoef(act[to_take], reg_2[to_take])[0, 1]-np.corrcoef(act[to_take], reg_1[to_take])[0, 1]
    return 1 - np.sum(all_d_corrs > 0) / nboot


def zscore(mat, axis=1):
    """
    ZScore matrix mat along given axis
    """
    m = np.mean(mat, axis, keepdims=True)
    s = np.std(mat, axis, keepdims=True)
    return (mat-m)/s


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
    labels = ["All motor", "Flicks", "Right flicks", "Left flicks", "Swims", "Stim only", "No stim"]

    # perform motor clustering and plot - use motor categories as well as stimulus dependence
    # slow POA ON regressor for finding heat-specific motor units at different timescales
    # NOTE: Initially tested with trigeminal ON activity as well but POA seems to recover more cells
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    poa_data = pickle.loads(np.array(storage["POA"]))  # type: RegionResults
    poa_sens = np.mean(poa_data.region_acts[poa_data.region_mem == 1, :], 0)
    storage.close()
    # we create the following stimulus-motor regressors:
    # A multiplication of mc_all with a 0/1 thresholded version of the activity trace (only events when active)
    poa_thresh = 0.25  # POA slow ON cells are considered active above this threshold, based on looking at average
    poa_on = poa_sens > 0.25
    poa_off = poa_sens <= 0.25
    motor_corrs = np.zeros((all_activity.shape[0], len(mc) + 2))
    for i, act in enumerate(all_activity):
        for j, m in enumerate(mc):
            motor_corrs[i, j] = np.corrcoef(act, m[i])[0, 1]
        motor_corrs[i, len(mc)] = np.corrcoef(act, mc_all[i] * poa_on)[0, 1]
        motor_corrs[i, len(mc) + 1] = np.corrcoef(act, mc_all[i] * poa_off)[0, 1]
    # find cells that have at least one above-threshold motor correlation
    motor_corrs[np.isnan(motor_corrs)] = 0
    mclust = MotorCluster(motor_corrs, mc, all_activity, poa_on, poa_off)
    motor_cells = (mclust.cluster_labels > -1)
    l_in_cluster = mclust.cluster_labels[motor_cells]
    motor_corrs_sig = motor_corrs[motor_cells, :].copy()

    # plot sorted by cluster identity
    fig, ax = pl.subplots()
    sns.heatmap(motor_corrs_sig[np.argsort(l_in_cluster), :], vmin=0, vmax=1, yticklabels=2500,
                xticklabels=labels, cmap="viridis")
    # plot cluster boundaries
    covered = 0
    for i in range(len(mc) + 2):
        covered += np.sum(l_in_cluster == i)
        ax.plot([0, motor_corrs_sig.shape[1] + 1], [l_in_cluster.size-covered, l_in_cluster.size-covered], 'k')
    ax.set_title('Above threshold correlations clustered and sorted')

    # create a membership array for plotting
    membership_motor = np.full_like(membership, -1)
    msh_nn = np.full_like(mship_nonan, -1)
    msh_nn[motor_cells] = l_in_cluster
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

    tailstore.close()

    # compute motor-type regressors
    # NOTE: Since each plane and therefore each cell only covers a fraction of motor events we can't simply average
    # dF/F values. This heavily biases the regressors to most active / brightest cells (SNR). Therefore sum z-scored
    # versions of the traces
    reg_all = np.mean(zscore(all_activity[mclust.cluster_labels == 0, :]), 0)
    is_flick_cell = np.logical_and(mclust.cluster_labels > 0, mclust.cluster_labels < 4)
    reg_flick = np.mean(zscore(all_activity[is_flick_cell, :]), 0)
    reg_swim = np.mean(zscore(all_activity[mclust.cluster_labels == 4, :]), 0)
    reg_stimOnly = np.mean(zscore(all_activity[mclust.cluster_labels == 5, :]), 0)
    reg_noStim = np.mean(zscore(all_activity[mclust.cluster_labels == 6, :]), 0)
    motor_type_regs = np.hstack(
        (reg_all[:, None], reg_flick[:, None], reg_swim[:, None], reg_stimOnly[:, None], reg_noStim[:, None]))
    # for later modeling, column normalize the regression matrix
    motor_type_regs = zscore(motor_type_regs, 0)
    swim_out = zscore(mc_swims.avg_motor_output, 0)
    flick_out = zscore(mc_flick.avg_motor_output, 0)

    # create a motor-system related file to store computed information
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "w-")
    motor_store.create_dataset("motor_corrs", data=motor_corrs)
    motor_store.create_dataset("cluster_labels", data=mclust.cluster_labels)
    motor_store.create_dataset("membership_motor", data=membership_motor)
    motor_store.create_dataset("motor_type_regs", data=motor_type_regs)
    motor_store.create_dataset("swim_out", data=swim_out)
    motor_store.create_dataset("flick_out", data=flick_out)
    motor_store.close()
