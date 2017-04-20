# Script to analyze region-specific contingencies between sensory activity and motor events
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import RegionContainer, assign_region_label, MotorContainer, SLHRepeatExperiment
from multiExpAnalysis import get_stack_types, dff, max_cluster
from typing import List
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D


def build_all_region_labels():
    """
    Creates numpy array with region label for all cells
    """
    c_main = tf_centroids[stack_types == "MAIN", :]
    c_tgl = tf_centroids[stack_types == "TG_LEFT", :]
    c_tgr = tf_centroids[stack_types == "TG_RIGHT", :]
    reg_main = assign_region_label(c_main, rl_main, 500/512/1.5)
    reg_tgl = assign_region_label(c_tgl, rl_tg_left, 500/512/2)
    reg_tgr = assign_region_label(c_tgr, rl_tg_right, 500/512/2)
    labels = []
    i_main = 0
    i_tgl = 0
    i_tgr = 0
    for st in stack_types:
        if st == "MAIN":
            labels.append(reg_main[i_main])
            i_main += 1
        elif st == "TG_LEFT":
            labels.append(reg_tgl[i_tgl])
            i_tgl += 1
        elif st == "TG_RIGHT":
            labels.append(reg_tgr[i_tgr])
            i_tgr += 1
        else:
            raise ValueError()
    return np.array(labels)


def activity_laterality(mc_left, mc_right, label_left, label_right, cluster_ids, fstd=0.5):
    """
    Computes motor-lateralized activity by brain region
    Args:
        mc_left: Motor container for "leftward" motor output
        mc_right: Motor container for "rightward" motor output
        label_left: Region label of the left side
        label_right: Region label of the right side
        cluster_ids: List of membership cluster numbers to include
        fstd: To reduce noise filter motor output with gaussian filter of this standard deviation

    Returns:
        [0]: Per cell average activity for ipsilateral motor events
        [1]: Per cell average activity for contralateral motor events
    """
    global all_rl, all_dff, mship_nonan
    ipsi = []
    contra = []
    cluster_member = np.zeros(mship_nonan.size, dtype=bool)
    for ci in cluster_ids:
        cluster_member = np.logical_or(cluster_member, mship_nonan == ci)
    ix_all = np.arange(all_dff.shape[0], dtype=int)
    ix_left = ix_all[np.logical_and(cluster_member, all_rl == label_left)]
    ix_right = ix_all[np.logical_and(cluster_member, all_rl == label_right)]
    for i in ix_left:
        l_starts = gaussian_filter1d(mc_left[i], fstd)
        r_starts = gaussian_filter1d(mc_right[i], fstd)
        trace = all_dff[i, :]
        if l_starts.sum() > 0:
            ipsi.append(np.sum(trace * l_starts) / l_starts.sum())
        if r_starts.sum() > 0:
            contra.append(np.sum(trace * r_starts) / r_starts.sum())
    for i in ix_right:
        l_starts = gaussian_filter1d(mc_left[i], fstd)
        r_starts = gaussian_filter1d(mc_right[i], fstd)
        trace = all_dff[i, :]
        if l_starts.sum() > 0:
            contra.append(np.sum(trace * l_starts) / l_starts.sum())
        if r_starts.sum() > 0:
            ipsi.append(np.sum(trace * r_starts) / r_starts.sum())
    return ipsi, contra


def sensory_mta(mc, region_labels, cluster_ids):
    """
    Computes movement triggered average of sensory activity
    Args:
        mc: 
        region_labels: 
        cluster_ids: 

    Returns:

    """
    pass


def motor_activity_boost(mc, region_labels, cluster_ids, fstd=0.5):
    """
    Computes difference of each cells activity during motor events to region average activity during motor events
    Args:
        mc: Motor container for considered motor output
        region_labels: List of region labels to analyze
        cluster_ids: List of cluster ids to consider
        fstd: To reduce noise filter motor output with gaussian filter of this standard deviation

    Returns:
        Per cell average activity during motor minus region average activity during motor
    """
    global all_rl, all_dff, mship_nonan
    act_boost = []
    cluster_member = np.zeros(mship_nonan.size, dtype=bool)
    if type(cluster_ids) == list:
        for ci in cluster_ids:
            cluster_member = np.logical_or(cluster_member, mship_nonan == ci)
    else:
        cluster_member = mship_nonan == cluster_ids
    region_member = np.zeros_like(cluster_member)
    if type(region_labels) == list:
        for rl in region_labels:
            region_member = np.logical_or(region_member, all_rl == rl)
    else:
        region_member = all_rl == region_labels
    ix_all = np.arange(all_dff.shape[0], dtype=int)
    ix_cons = ix_all[np.logical_and(cluster_member, region_member)]
    avg_activity = np.mean(all_dff[ix_cons, :], 0)
    for i in ix_cons:
        starts = mc[i]
        trace = all_dff[i, :] - avg_activity
        if starts.sum() > 0:
            act_boost.append(np.sum(trace * starts) / starts.sum())
    return act_boost


def trial_average(ts):
    if ts.ndim == 1:
        return np.mean(ts.reshape((3, ts.size//3)), 0)
    elif ts.ndim == 2:
        return np.mean(ts.reshape((ts.shape[0], 3, ts.shape[1]//3)), 1)


def build_region_clusters(region_labels, n_regs=6, plot=True):
    """
    Clusters cells in inidicated regions by performing spectral clustering followed by regression-based
    identification with r>=0.6 cut-off
    Args:
        region_labels: A list of labels of regions to include
        n_regs: The number of regressors to extract
        plot: If set to true, plot regressors, regressor correlation matrix and embedding

    Returns:
        [0]: Activity of all cells that are part of the indicated regions
        [1]: Cluster membership of each cell
    """
    global all_rl, all_dff, mship_nonan
    region_member = np.zeros_like(mship_nonan.size, dtype=bool)
    if type(region_labels) == list:
        for rl in region_labels:
            region_member = np.logical_or(region_member, all_rl == rl)
    else:
        region_member = all_rl == region_labels
    region_activity = all_dff[region_member, :]
    to_cluster = trial_average(region_activity)
    corr_mat = np.corrcoef(to_cluster)
    corr_mat[corr_mat < 0.26] = 0
    spc_clust = SpectralClustering(n_clusters=n_regs, affinity="precomputed")
    spec_embed = SpectralEmbedding(n_components=3, affinity='precomputed')
    cids = spc_clust.fit_predict(corr_mat)
    coords = spec_embed.fit_transform(corr_mat)
    spec_regs = np.zeros((region_activity.shape[1], n_regs))
    for i in range(n_regs):
        spec_regs[:, i] = np.mean(region_activity[cids == i, :], 0)
    # for unified sorting determine correlation of each regressor to our "on stimulus"
    c = np.array([np.corrcoef(spec_regs[:, i], exp_data[0].stimOn)[0, 1] for i in range(n_regs)])
    spec_regs = spec_regs[:, np.argsort(c)]
    reg_corr_mat = np.zeros((region_activity.shape[0], n_regs))
    for i in range(region_activity.shape[0]):
        for j in range(n_regs):
            reg_corr_mat[i, j] = np.corrcoef(region_activity[i, :], spec_regs[:, j])[0, 1]
    reg_corr_mat[np.abs(reg_corr_mat) < 0.6] = 0
    ab_thresh = np.sum(reg_corr_mat > 0, 1) > 0
    # remove all cells that don't have above-threshold correlations and assign to cluster based on max correlation
    reg_corr_mat = reg_corr_mat[ab_thresh, :].copy()
    km = max_cluster(np.nanargmax(reg_corr_mat, 1))
    if plot:
        # plot regressor correlation matrix of clustered cells
        fig, ax = pl.subplots()
        sns.heatmap(reg_corr_mat[np.argsort(km.labels_), :], yticklabels=reg_corr_mat.shape[0]//10, ax=ax)
        covered = 0
        for i in range(km.n_clusters):
            covered += np.sum(km.labels_ == i)
            ax.plot([0, n_regs+1], [km.labels_.size-covered, km.labels_.size-covered], 'k')
    membership = np.full(region_activity.shape[0], -1, dtype=int)
    membership[ab_thresh] = km.labels_.astype(int)
    # don't allow clusters of singular traces
    for i in range(n_regs):
        if np.sum(membership == i) < 2:
            membership[membership == i] = -1
    if plot:
        # plot per-trial cluster average traces
        fig, ax = pl.subplots()
        for i in range(n_regs):
            # ax.plot(spec_regs[:, i])
            ax.plot(np.mean(trial_average(region_activity[membership == i, :]), 0))
        sns.despine(fig, ax)
        # plot cluster members in embedded space
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n_regs):
                ax.scatter(coords[membership == i, 0], coords[membership == i, 1],
                           coords[membership == i, 2], s=5)
    return region_activity, membership


def build_regressors(activity, cluster_membership):
    """
    For each cluster value in cluster_membership creates a regressor corresponding to the average trace
    of that cluster
    Args:
        activity: Activity matrix
        cluster_membership: Membership of each cell/row in the activity matrix

    Returns:
        n_timepoints x n_clusters sized regressor matrix
    """
    n_regs = np.unique(cluster_membership[cluster_membership != -1]).size
    regressors = np.zeros((activity.shape[1], n_regs))
    for i, c in enumerate(np.unique(cluster_membership[cluster_membership != -1])):
        regressors[:, i] = np.mean(activity[cluster_membership == c, :], 0)
    return regressors


if __name__ == "__main__":
    sns.reset_orig()
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    all_activity = np.array(dfile['all_activity'])
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
    # load segmentation files
    segFile = h5py.File("E:/Dropbox/2P_FuncStack_Annotation/fullStack_segmentation.hdf5")
    rl_main = RegionContainer.load_container_list(segFile)
    segFile.close()
    segFile = h5py.File("E:/Dropbox/2P_FuncStack_Annotation/TG_Left_Segmentation.hdf5")
    rl_tg_left = RegionContainer.load_container_list(segFile)
    segFile.close()
    segFile = h5py.File("E:/Dropbox/2P_FuncStack_Annotation/TG_Right_Segmentation.hdf5")
    rl_tg_right = RegionContainer.load_container_list(segFile)
    segFile.close()

    # create motor containers
    itime = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    mc_all_raw = MotorContainer(sourceFiles, itime, 0)
    mc_flicks_raw = MotorContainer(sourceFiles, itime, 0, predicate=high_bias_bouts, tdd=mc_all_raw.tdd)
    mc_flick_left_raw = MotorContainer(sourceFiles, itime, 0, predicate=left_bias_bouts, tdd=mc_all_raw.tdd)
    mc_flick_right_raw = MotorContainer(sourceFiles, itime, 0, predicate=right_bias_bouts, tdd=mc_all_raw.tdd)
    stack_types = get_stack_types(exp_data)[no_nan_aa]
    # get indices of on-type cells in regions (since everything is limited to no_nan_aa these should match)
    all_rl = build_all_region_labels()
    ix_all = np.arange(all_activity.shape[0], dtype=int)
    all_dff = dff(all_activity)

    ##################
    # Analyze relation of lateralized activity and motor laterality
    ##################
    # ON cell clusters
    on_c = [0, 1, 2]
    ipsi_tg_on, contra_tg_on = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "TG_L", "TG_R", on_c)
    ipsi_rh6_on, contra_rh6_on = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "Rh6_L", "Rh6_R", on_c)
    ipsi_cer_on, contra_cer_on = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "Cerebellum_L", "Cerebellum_R", on_c)
    ipsi_hab_on, contra_hab_on = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "Hab_L", "Hab_R", on_c)

    # OFF cell clusters
    off_c = [4, 5]
    ipsi_tg_off, contra_tg_off = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "TG_L", "TG_R", off_c)
    ipsi_rh6_off, contra_rh6_off = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "Rh6_L", "Rh6_R", off_c)
    ipsi_cer_off, contra_cer_off = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "Cerebellum_L", "Cerebellum_R", off_c)
    ipsi_hab_off, contra_hab_off = activity_laterality(mc_flick_left_raw, mc_flick_right_raw, "Hab_L", "Hab_R", off_c)

    fig, (ax_on, ax_off) = pl.subplots(nrows=2, ncols=4, sharey=True)
    # plot on
    sns.barplot(data=(ipsi_tg_on, contra_tg_on), ax=ax_on[0])
    ax_on[0].set_title("Trigeminal")
    ax_on[0].set_ylabel("$\Delta$F/F0 ON")
    sns.barplot(data=(ipsi_rh6_on, contra_rh6_on), ax=ax_on[1])
    ax_on[1].set_title("Rh6 cluster")
    sns.barplot(data=(ipsi_cer_on, contra_cer_on), ax=ax_on[2])
    ax_on[2].set_title("Cerebellum")
    sns.barplot(data=(ipsi_hab_on, contra_hab_on), ax=ax_on[3])
    ax_on[3].set_title("Habenula")
    # plot off
    sns.barplot(data=(ipsi_tg_off, contra_tg_off), ax=ax_off[0])
    ax_off[0].set_ylabel("$\Delta$F/F0 OFF")
    sns.barplot(data=(ipsi_rh6_off, contra_rh6_off), ax=ax_off[1])
    sns.barplot(data=(ipsi_cer_off, contra_cer_off), ax=ax_off[2])
    sns.barplot(data=(ipsi_hab_off, contra_hab_off), ax=ax_off[3])
    sns.despine(fig)
    fig.tight_layout()

    ##################
    # Analyze predictability of average motor outcomes (convolved) by cluster activity in different regions
    ##################
    cat = exp_data[0].caTimeConstant
    mc_all = MotorContainer(sourceFiles, itime, cat, tdd=mc_all_raw.tdd)
    mc_flicks = MotorContainer(sourceFiles, itime, cat, predicate=high_bias_bouts, tdd=mc_all_raw.tdd)
    mc_swims = MotorContainer(sourceFiles, itime, cat, predicate=unbiased_bouts, tdd=mc_all_raw.tdd)

    time = np.arange(all_dff.shape[1]) / 5

    all_motor = trial_average(mc_all.avg_motor_output)
    flicks_motor = trial_average(mc_flicks.avg_motor_output)
    swim_motor = trial_average(mc_swims.avg_motor_output)

    # example region derivation
    regions = ["TG_L", "TG_R"]
    region_act, region_mem = build_region_clusters(regions)

    regressors = build_regressors(trial_average(region_act), region_mem)

    resmat_high = np.zeros((regressors.shape[1], regressors.shape[1]))
    resmat_low = np.zeros_like(resmat_high)
    resmat_all = np.zeros_like(resmat_high)
    for i in range(regressors.shape[1]):
        for j in range(regressors.shape[1]):
            if j < i:
                resmat_high[i, j] = np.nan
                resmat_low[i, j] = np.nan
                resmat_all[i, j] = np.nan
                continue
            regs = np.hstack((regressors[:, i, None], regressors[:, j, None]))
            lreg = LinearRegression()
            lreg.fit(regs, flicks_motor)
            resmat_high[i, j] = lreg.score(regs, flicks_motor)
            lreg = LinearRegression()
            lreg.fit(regs, swim_motor)
            resmat_low[i, j] = lreg.score(regs, swim_motor)
            lreg = LinearRegression()
            lreg.fit(regs, all_motor)
            resmat_all[i, j] = lreg.score(regs, all_motor)

    # Plot prediction of general motor output
    fig, ax = pl.subplots()
    sns.heatmap(resmat_all, 0, 1, cmap="RdBu_r", annot=True, ax=ax)
    ax.set_title("Prediction of overall motor output")

    # Plot prediction of low and high bias side-by-sid
    fig, (ax_h, ax_l) = pl.subplots(ncols=2)
    sns.heatmap(resmat_high, 0, 1, cmap="RdBu_r", annot=True, ax=ax_h)
    ax_h.set_title("Prediction of strong flicks")
    sns.heatmap(resmat_low, 0, 1, cmap="RdBu_r", annot=True, ax=ax_l)
    ax_l.set_title("Prediction of swims")
    fig.tight_layout()
