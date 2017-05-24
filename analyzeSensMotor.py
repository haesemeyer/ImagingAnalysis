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
import matplotlib as mpl
import pandas
from scipy.stats import entropy


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


def build_region_clusters(region_labels, n_regs=6, plot=True, corr_cut_off=0.9, plTitle=""):
    """
    Clusters cells in inidicated regions by performing spectral clustering followed by regression-based
    identification with r>=0.6 cut-off
    Args:
        region_labels: A list of labels of regions to include
        n_regs: The number of regressors to extract
        plot: If set to true, plot regressors, regressor correlation matrix and embedding
        corr_cut_off: If two clusters are at least this correlated by avg. activity they will be merged
        plTitle: For id purposes title of plots

    Returns:
        [0]: Activity of all cells that are part of the indicated regions
        [1]: Cluster membership of each cell
        [3]: Original indexes into all_dff of the cells that are part of the selected regions for re-mapping
    """
    global all_rl, all_dff, mship_nonan
    region_member = np.zeros_like(mship_nonan.size, dtype=bool)
    if type(region_labels) == list:
        for rl in region_labels:
            region_member = np.logical_or(region_member, all_rl == rl)
    else:
        region_member = all_rl == region_labels
    region_activity = all_dff[region_member, :]
    region_indices = np.arange(all_dff.shape[0], dtype=int)[region_member]
    to_cluster = trial_average(region_activity)
    print("{0} cells to process in this run".format(to_cluster.shape[0]))
    ix = np.arange(to_cluster.shape[0])
    if to_cluster.shape[0] > 30000:
        # subsample
        ix = np.random.choice(ix, 30000, False)
        to_cluster = to_cluster[ix, :]
        print("Performing subsampling to 30.000 cells")
    corr_mat = np.corrcoef(to_cluster).astype(np.float32)
    del to_cluster
    corr_mat[corr_mat < 0.26] = 0
    spc_clust = SpectralClustering(n_clusters=n_regs, affinity="precomputed")
    # spec_embed = SpectralEmbedding(n_components=3, affinity='precomputed')
    cids = spc_clust.fit_predict(corr_mat)
    # coords = spec_embed.fit_transform(corr_mat)
    spec_regs = np.zeros((region_activity.shape[1], n_regs))
    for i in range(n_regs):
        spec_regs[:, i] = np.mean(region_activity[ix, :][cids == i, :], 0)
    # for unified sorting determine correlation of each regressor to our "on stimulus"
    c = np.array([np.corrcoef(spec_regs[:, i], exp_data[0].stimOn)[0, 1] for i in range(n_regs)])
    spec_regs = spec_regs[:, np.argsort(-1*c)]  # sort by descending correlation
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
        if plTitle != "":
            ax.set_title(plTitle)
    membership = np.full(region_activity.shape[0], -1, dtype=int)
    membership[ab_thresh] = km.labels_.astype(int)
    # don't allow clusters of less than 5 constituents
    for i in range(n_regs):
        if np.sum(membership == i) < 5:
            membership[membership == i] = -1
    # merge clusters with a correlation >= corr_cut_off
    avgs, clust_labels = build_regressors(trial_average(region_activity), membership)
    avg__avg_corrs = np.corrcoef(avgs.T)
    for i in range(avgs.shape[1]):
        for j in range(i+1, avgs.shape[1]):
            if avg__avg_corrs[i, j] >= corr_cut_off:
                membership[membership == clust_labels[j]] = clust_labels[i]
    if plot:
        # plot per-trial cluster average traces
        fig, ax = pl.subplots()
        time = np.arange(region_activity.shape[1]//3) / 5
        for i in range(n_regs):
            if np.sum(membership == i) > 0:
                ax.plot(time, np.mean(trial_average(region_activity[membership == i, :]), 0), label=str(i))
        if plTitle != "":
            ax.set_title(plTitle)
        ax.legend()
        sns.despine(fig, ax)
        # plot cluster members in embedded space
        # fig = pl.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(n_regs):
        #     if np.sum(membership == i) > 0:
        #         ax.scatter(coords[membership == i, 0], coords[membership == i, 1],
        #                    coords[membership == i, 2], s=5)
        # if plTitle != "":
        #     ax.set_title(plTitle)
    return region_activity, membership, region_indices


def build_regressors(activity, cluster_membership):
    """
    For each cluster value in cluster_membership creates a regressor corresponding to the average trace
    of that cluster
    Args:
        activity: Activity matrix
        cluster_membership: Membership of each cell/row in the activity matrix

    Returns:
        n_timepoints x n_clusters sized regressor matrix
        n_clusters sized vector with the memberhip label of each cluster
    """
    n_regs = np.unique(cluster_membership[cluster_membership != -1]).size
    regressors = np.zeros((activity.shape[1], n_regs))
    mem_label = []
    for i, c in enumerate(np.unique(cluster_membership[cluster_membership != -1])):
        regressors[:, i] = np.mean(activity[cluster_membership == c, :], 0)
        mem_label.append(c)
    return regressors, np.array(mem_label)


def regression_CV(regressors: np.ndarray, output: np.ndarray, nboot=500):
    """
    Performs 80/20 cross-validation of regression leaving out 20% contiguous percent of the data for fitting.
    This avoids having each left-out data-point ensconced btw. two highly correlated fitted points
    Args:
        regressors: The regressors to use
        output: The output to predict
        nboot: The number of cross-validations to perform

    Returns:
        [0]: nboot R2 values of the cross validation
        [1]: nbootxregressors.shape[1] matrix of coefficients
        [2]: nboot intercept values
    """
    def wrap_indices(start, length, size):
        """
        Returns a series of length indices starting at index start and wrapping around to 0 according to size
        """
        return np.sort(np.arange(start, start+length) % size)

    if output.ndim < 2:
        output = output[:, None]
    lo_stretch_length = regressors.shape[0] // 5
    r2_vals = np.zeros(nboot)
    coefs = np.zeros((nboot, output.shape[1], regressors.shape[1]))
    icepts = np.zeros((nboot, output.shape[1]))
    all_ix = np.arange(regressors.shape[0])
    for i in range(nboot):
        lo_start = np.random.randint(regressors.shape[0])
        ix_leave_out = wrap_indices(lo_start, lo_stretch_length, regressors.shape[0])
        ix_fit = np.setdiff1d(all_ix, ix_leave_out)
        lreg = LinearRegression()
        lreg.fit(regressors[ix_fit, :], output[ix_fit, :])
        coefs[i, :, :] = lreg.coef_
        icepts[i, :] = lreg.intercept_
        r2_vals[i] = lreg.score(regressors[ix_leave_out, :], output[ix_leave_out, :])
    return r2_vals, coefs, icepts


def regression_bootstrap(regressors: np.ndarray, output: np.ndarray, nboot=500):
    """
        Performs bootstrapping of regression model
        Args:
            regressors: The regressors to use
            output: The output to predict
            nboot: The number of bootstraps to perform

        Returns:
            [0]: nboot R2 values of the fits
            [1]: nbootxregressors.shape[1] matrix of coefficients
            [2]: nboot intercept values
    """
    if output.ndim < 2:
        output = output[:, None]
    if regressors.ndim < 2:
        regressors = regressors[:, None]
    r2_vals = np.zeros(nboot)
    coefs = np.zeros((nboot, output.shape[1], regressors.shape[1]))
    icepts = np.zeros((nboot, output.shape[1]))
    all_ix = np.arange(regressors.shape[0])
    for i in range(nboot):
        to_take = np.random.choice(all_ix, all_ix.size, True)
        lreg = LinearRegression()
        lreg.fit(regressors[to_take, :], output[to_take, :])
        coefs[i, :, :] = lreg.coef_
        icepts[i, :] = lreg.intercept_
        r2_vals[i] = lreg.score(regressors[to_take, :], output[to_take, :])
    return r2_vals, coefs, icepts


def jknife_entropy(data, nbins=10):
    """
    Computes the jacknife estimate of the entropy of data (Zahl, 1977)
    Args:
        data: nsamples x ndim sized data matrix
        nbins: The number of histogram bins to use along each dimension

    Returns:
        The jacknife estimate of the entropy
    """
    hist = np.histogramdd(data, nbins)[0].ravel()
    ent_full = entropy(hist)
    # jacknife
    jk_sum = 0
    jk_n = 0
    for i in range(hist.size):
        if hist[i] > 0:
            jk_hist = hist.copy()
            jk_hist[i] -= 1
            # for each element in this bin we get exactly one jack-nife estimate
            jk_sum = jk_sum + hist[i] * entropy(jk_hist)
            jk_n += hist[i]
    return hist.sum() * ent_full - (hist.sum() - 1)*jk_sum/jk_n


class RegionResults:
    def __init__(self, name, activities, membership, regressors, original_labels):
        self.name = name
        self.region_acts = activities
        self.region_mem = membership
        self.regressors = regressors
        self.regs_clust_labels = original_labels
        self.full_averages = None


if __name__ == "__main__":
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    all_activity = np.array(dfile['all_activity'])
    # # rotate each line in all_activity
    # for i in range(all_activity.shape[0]):
    #     r = np.random.randint(30 * 5, 120 * 5)
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
    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    itime = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    mc_all_raw = MotorContainer(sourceFiles, itime, 0, hdf5_store=tailstore)
    mc_flicks_raw = MotorContainer(sourceFiles, itime, 0, predicate=high_bias_bouts, tdd=mc_all_raw.tdd)
    mc_flick_left_raw = MotorContainer(sourceFiles, itime, 0, predicate=left_bias_bouts, tdd=mc_all_raw.tdd)
    mc_flick_right_raw = MotorContainer(sourceFiles, itime, 0, predicate=right_bias_bouts, tdd=mc_all_raw.tdd)
    stack_types = get_stack_types(exp_data)[no_nan_aa]
    # get indices of on-type cells in regions (since everything is limited to no_nan_aa these should match)
    all_rl = build_all_region_labels()
    # remove all cells that had been classified as motor from further analysis by blanking their region labels
    all_rl[mship_nonan > 5] = ""
    ix_all = np.arange(all_activity.shape[0], dtype=int)
    all_dff = dff(all_activity)

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

    # plot motor type output
    trial_time = np.arange(flicks_motor.size) / 5
    fig, ax = pl.subplots()
    ax.plot(trial_time, swim_motor, label="Swims")
    ax.plot(trial_time, flicks_motor, label="Flicks")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Probability")
    sns.despine(fig, ax)

    test_regions = [["TG_L", "TG_R"],
                    ["Rh6_L", "Rh6_R"],
                    ["Rh2_L", "Rh2_R"],
                    ["Cerebellum_L", "Cerebellum_R"],
                    ["Hab_L", "Hab_R"],
                    ["D_FB_L", "D_FB_R"],
                    "SubPallium",
                    "PreOptic"
                    ]

    test_labels = ["Trigeminal", "Rh6", "Rh2", "Cerebellum", "Habenula", "Pallium", "SubPallium", "POA"]

    motor_out = np.hstack((swim_motor[:, None], flicks_motor[:, None]))
    # for entropy use full timeseries not repeat averages
    mo = np.hstack((mc_swims.avg_motor_output[:, None], mc_flicks.avg_motor_output[:, None]))
    mo_entropy = jknife_entropy(mo, 10)
    n_boot = 500
    region_r2 = np.zeros((n_boot, len(test_labels)))
    region_boot_coefs = []  # for each region array of bootstrap coefficients (n_bootxn_regsx2)
    region_boot_icepts = []  # for each region matrix of bootstrap intercepts (n_bootx2)
    region_mi = np.zeros(len(test_labels))

    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r+')

    for k, regions in enumerate(test_regions):
        print("Processing ", test_labels[k])
        region_act, region_mem = build_region_clusters(regions, plTitle=test_labels[k])[:2]
        regressors, clust_labels = build_regressors(trial_average(region_act), region_mem)
        full_averages = build_regressors(region_act, region_mem)[0]
        # create prediction matrices
        resmat_high = np.zeros((regressors.shape[1], regressors.shape[1]))
        resmat_low = np.zeros_like(resmat_high)
        for i in range(regressors.shape[1]):
            for j in range(regressors.shape[1]):
                if j < i:
                    resmat_high[i, j] = np.nan
                    resmat_low[i, j] = np.nan
                    continue
                regs = np.hstack((full_averages[:, i, None], full_averages[:, j, None]))
                lreg = LinearRegression()
                lreg.fit(regs, mo[:, 1])
                resmat_high[i, j] = lreg.score(regs, mo[:, 1])
                lreg = LinearRegression()
                lreg.fit(regs, mo[:, 0])
                resmat_low[i, j] = lreg.score(regs, mo[:, 0])
                # lreg = LinearRegression()
                # lreg.fit(regs, all_motor)
        # Plot prediction of low and high bias side-by-side
        fig, (ax_h, ax_l) = pl.subplots(ncols=2)
        sns.heatmap(resmat_high, 0, 1, cmap="RdBu_r", annot=True, ax=ax_h, xticklabels=clust_labels,
                    yticklabels=clust_labels)
        ax_h.set_title("Prediction of strong flicks")
        sns.heatmap(resmat_low, 0, 1, cmap="RdBu_r", annot=True, ax=ax_l, xticklabels=clust_labels,
                    yticklabels=clust_labels)
        ax_l.set_title("Prediction of swims {0}".format(test_labels[k]))
        fig.tight_layout()

        # plot regressor-regressor correlations
        fig, ax = pl.subplots()
        sns.heatmap(np.corrcoef(regressors.T), vmax=1, annot=True, xticklabels=clust_labels,
                    yticklabels=clust_labels, ax=ax)
        ax.set_title(test_labels[k])
        analysis_result = RegionResults(test_labels[k], region_act, region_mem, regressors, clust_labels)
        analysis_result.full_averages = full_averages
        # compute regression results on full data not repeat-averaged
        r2, coef, icepts = regression_CV(full_averages, mo, n_boot)
        region_boot_coefs.append(coef)
        region_boot_icepts.append(coef)
        region_r2[:, k] = r2
        # compute mutual information of region regressors with motor output
        region_entropy = jknife_entropy(full_averages, 10)
        joint_entropy = jknife_entropy(np.hstack((full_averages, mo)), 10)
        region_mi[k] = mo_entropy + region_entropy - joint_entropy

        # save new analysis into our file
        l = test_labels[k]
        if l in storage:
            del storage[l]
        stream = pickle.dumps(analysis_result)
        storage.create_dataset(l, data=np.void(stream))
        # flush new data to disk
        storage.flush()
        del stream
        del analysis_result

        print("{0} out of {1} done.".format(k+1, len(test_regions)))
    # close storage file to save remaining data then re-open read-only
    storage.close()
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    # plot CV r-squared across regions
    dset = pandas.DataFrame({test_labels[k]: region_r2[:, k] for k in range(len(test_labels))})
    fig, ax = pl.subplots()
    sns.barplot(data=dset, ax=ax, order=test_labels, ci=68)
    ax.set_ylabel("Cross-validation R2")
    sns.despine(fig, ax)

    # replot regressors with error shadings
    for k in test_labels:
        fig, ax = pl.subplots()
        ar = pickle.loads(np.array(storage[k]))
        for c in ar.regs_clust_labels:
            cname = "C{0}".format(int(c))
            data = trial_average(ar.region_acts[ar.region_mem == c, :])
            sns.tsplot(data=data, time=trial_time, color=cname, ax=ax)
        sns.despine(fig, ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("dF/F0")
        ax.set_title(k)
    storage.close()
