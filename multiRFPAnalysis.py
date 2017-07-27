from mh_2P import OpenStack, TailData, UiGetFile, NucGraph, CorrelationGraph, SOORepeatExperiment
from mh_2P import vec_mat_corr, KDTree
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from time import perf_counter
import scipy.signal as signal
import pickle
import nrrd
import sys
import os
import subprocess as sp
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from multiExpAnalysis import max_cluster, n_r2_above_thresh, n_exp_r2_above_thresh, dff
from Figure1 import trial_average

sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')
from mhba_basic import Crosscorrelation


def DumpAnalysisDataHdf5(filename):
    """
    Save all created analysis data into one hdf5 file for easy re-analysis
    Args:
        filename: The name of the file to create - if file already exists exception will be raised
    Returns:
        True on success
    """
    import h5py
    try:
        dfile = h5py.File(filename, 'w-')  # fail if file already exists
    except OSError:
        print("Could not create file. Make sure a file with the same name doesn't exist in this location.")
        return False
    try:
        # save major data structures compressed
        dfile.create_dataset("all_activity", data=all_activity, compression="gzip", compression_opts=9)
        dfile.create_dataset("avg_analysis_data", data=avg_analysis_data, compression="gzip", compression_opts=9)
        # pickle our experiment data as a byte-stream
        pstring = pickle.dumps(exp_data)
        dfile.create_dataset("exp_data_pickle", data=np.void(pstring))
        # save smaller arrays uncompressed
        dfile.create_dataset("exp_id", data=exp_id)
        dfile.create_dataset("membership", data=membership)
        dfile.create_dataset("no_nan", data=no_nan)
        dfile.create_dataset("no_nan_aa", data=no_nan_aa)
        dfile.create_dataset("reg_corr_mat", data=reg_corr_mat)
        dfile.create_dataset("reg_trans", data=reg_trans)
    finally:
        dfile.close()
    return True


def printElapsed():
    elapsed = perf_counter() - t_start
    print(str(elapsed) + " s elapsed since start.", flush=True)


if __name__ == "__main__":
    print("Load data files", flush=True)
    exp_data = []
    dfnames = UiGetFile([('Experiment data', '.pickle')], True)
    t_start = perf_counter()
    for name in dfnames:
        f = open(name, 'rb')
        d = pickle.load(f)
        exp_data.append(d)
    is_pot_stim = np.array([], dtype=np.bool)  # for each unit whether it is potentially a stimulus driven unit
    exp_id = np.array([])  # for each unit the experiment which it came from
    stim_phase = np.array([])  # for each unit the phase at stimulus frequency during the sine-presentation
    all_activity = np.array([])
    source_files = []  # for each unit the imaging file from which it was derived
    for i, data in enumerate(exp_data):
        stim_fluct = data.computeStimulusEffect(0)[0].ravel()
        exp_id = np.r_[exp_id, np.full(stim_fluct.size, i, np.int32)]
        ips = stim_fluct >= 1
        is_pot_stim = np.r_[is_pot_stim, ips]
        if i == 0:
            # NOTE: RawData field is 64 bit float - convert to 32 when storing in all_activity
            all_activity = data.RawData.astype(np.float32)
        else:
            all_activity = np.r_[all_activity, data.RawData.astype(np.float32)]
        # after this point, the raw-data and vigor fields of experiment data are unecessary
        data.RawData = None
        data.Vigor = None
        source_files += [g[0] for g in data.graph_info]
    print("Data loaded and aggregated", flush=True)
    printElapsed()

    # Filter our activity traces with a gaussian window for rate estimation - sigma = 0.3 seconds
    window_size = 0.3 * exp_data[0].frameRate
    all_activity = gaussian_filter1d(all_activity, window_size, axis=1)
    # Plot the frequency gain characteristics of the filter used
    delta = np.zeros(21)
    delta[10] = 1
    filter_coeff = gaussian_filter1d(delta, window_size)
    nyquist = exp_data[0].frameRate/2
    w, h = signal.freqz(filter_coeff, worN=8000)
    with sns.axes_style('whitegrid'):
        pl.figure()
        pl.plot((w/np.pi)*nyquist, np.absolute(h), linewidth=2)
        pl.xlabel('Frequency [Hz]')
        pl.ylabel('Gain')
        pl.title('Frequency response of gaussian filter')
        pl.ylim(-0.05, 1.05)
    print("Activity filtered", flush=True)
    printElapsed()

    # for each cell that passes is_pot_stim holds the count of the number of other cells and other experiments that
    # are above the correlation threshold
    n_cells_above = np.zeros(is_pot_stim.sum(), dtype=np.int32)
    n_exp_above = np.zeros(is_pot_stim.sum(), dtype=np.int32)

    filter_r2_thresh = 0.4

    # make a temporary copy of all stimulus relevant units
    stim_act = all_activity[is_pot_stim, :].copy()
    rowmean = np.mean(stim_act, 1, keepdims=True)
    stim_act -= rowmean
    norms = np.linalg.norm(stim_act, axis=1)

    for i, row in enumerate(all_activity[is_pot_stim, :]):
        corrs = vec_mat_corr(row, stim_act, False, vnorm=norms[i], mnorm=norms)
        n_cells_above[i] = n_r2_above_thresh(corrs, filter_r2_thresh)
        n_exp_above[i] = n_exp_r2_above_thresh(corrs, filter_r2_thresh, exp_id[is_pot_stim])

    del stim_act

    # generate plot of number of cells to analyze for different "other cell" and "n experiment" criteria
    n_exp_to_test = [0, 1, 2, 3, 4, 5]
    n_cells_to_test = list(range(30))
    exp_above = np.zeros((len(n_exp_to_test), n_cells_above.size))
    cells_above = np.zeros((len(n_cells_to_test), n_cells_above.size))
    for i, net in enumerate(n_exp_to_test):
        exp_above[i, :] = n_exp_above >= net
    for i, nct in enumerate(n_cells_to_test):
        cells_above[i, :] = n_cells_above >= nct

    with sns.axes_style('whitegrid'):
        fig, ax = pl.subplots()
        for i, net in enumerate(n_exp_to_test):
            n_remain = []
            for j, nct in enumerate(n_cells_to_test):
                n_remain.append(np.sum(np.logical_and(exp_above[i, :], cells_above[j, :])) / n_cells_above.size)
            ax.plot(n_cells_to_test, n_remain, 'o', label="At least "+str(net)+" other fish")
        ax.set_xlabel('Number of other cells with $R^2$ > ' + str(filter_r2_thresh))
        ax.set_ylabel('Fraction of stimulus cells to analyze')
        ax.set_xlim(-0.2)
        ax.set_yscale('log')
        ax.legend()

    # get all cells that have at least 20 other cells with a timeseries R2>0.5
    # since there are much fewer RFP experiments only ask for presence in one other fish
    exp_g_1 = n_exp_above > 0
    c_g_9 = n_cells_above > 19
    to_analyze = np.logical_and(exp_g_1, c_g_9)
    analysis_data = all_activity[is_pot_stim, :][to_analyze, :]
    # marks, which of all units where used to derive regressors
    discovery_unit_marker = np.zeros(all_activity.shape[0], dtype=bool)
    discovery_unit_marker[is_pot_stim] = to_analyze

    # remove NaN containing traces from activity and motor matrix
    no_nan_aa = np.sum(np.logical_or(np.isnan(all_activity), np.isinf(all_activity)), 1) == 0
    all_activity = all_activity[no_nan_aa, :]
    discovery_unit_marker = discovery_unit_marker[no_nan_aa]
    print("Data filtering complete", flush=True)
    printElapsed()

    # TODO: The following is experiment specific - needs cleanup!
    avg_analysis_data = np.zeros((analysis_data.shape[0], analysis_data.shape[1] // 2))
    avg_analysis_data += analysis_data[:, :825]
    avg_analysis_data += analysis_data[:, 825:825*2]
    # use spectral clustering to identify our regressors. Use thresholded correlations as the affinity
    aad_corrs = np.corrcoef(avg_analysis_data)
    aad_corrs[aad_corrs < 0.26] = 0  # 99th percentile of correlations of a *column* shuffle of avg_analysis_data
    n_regs = 6  # extract 6 clusters as regressors
    spec_embed = SpectralEmbedding(n_components=3, affinity='precomputed')
    spec_clust = SpectralClustering(n_clusters=n_regs, affinity='precomputed')
    spec_embed_coords = spec_embed.fit_transform(aad_corrs)
    spec_clust_ids = spec_clust.fit_predict(aad_corrs)

    reg_orig = np.empty((analysis_data.shape[1], n_regs))
    time = np.arange(reg_orig.shape[0]) / 5
    for i in range(n_regs):
        reg_orig[:, i] = np.mean(dff(analysis_data[spec_clust_ids == i, :]), 0)
    # copy regressors and sort by correlation to data.stimOn - since we expect this to be most prevalent group
    reg_trans = reg_orig.copy()
    c = np.array([np.corrcoef(reg_orig[:, i], data.stimOn)[0, 1] for i in range(n_regs)])

    # make 3D plot of clusters in embedded space
    n_on = np.sum(c > 0)
    n_off = n_regs - n_on
    cols_on = sns.palettes.color_palette('bright', n_on)
    cols_off = sns.palettes.color_palette('deep', n_off)
    count_on = 0
    count_off = 0
    with sns.axes_style('white'):
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n_regs):
            if c[i] > 0:
                ax.scatter(spec_embed_coords[spec_clust_ids == i, 0], spec_embed_coords[spec_clust_ids == i, 1],
                           spec_embed_coords[spec_clust_ids == i, 2], s=5, c=cols_on[count_on])
                count_on += 1
            else:
                ax.scatter(spec_embed_coords[spec_clust_ids == i, 0], spec_embed_coords[spec_clust_ids == i, 1],
                           spec_embed_coords[spec_clust_ids == i, 2], s=5, c=cols_off[count_off])
                count_off += 1

    reg_trans = reg_trans[:, np.argsort(c)[::-1]]
    c = np.sort(c)[::-1]

    # plot our on and off type clusters in two different plots
    with sns.axes_style('whitegrid'):
        fig, ax_on = pl.subplots()
        fig, ax_off = pl.subplots()
        for i in range(n_regs):
            lab = 'Regressor ' + str(i)
            if c[i] > 0:
                ax_on.plot(time, reg_trans[:, i], label=lab, c=cols_on[i])
            else:
                ax_off.plot(time, reg_trans[:, i], label=lab, c=cols_off[i-n_on])
        # ax_on.legend(loc=2)
        # ax_off.legend(loc=2)
        ax_on.set_title('ON type regressors')
        ax_on.set_xlabel('Time [s]')
        ax_on.set_ylabel('dF/ F0')
        ax_off.set_title('OFF type regressors')
        ax_off.set_xlabel('Time [s]')
        ax_off.set_ylabel('dF/ F0')
    print("Stimulus regressor derivation complete", flush=True)
    printElapsed()

    # create matrix, that for each unit contains its correlation to each regressor
    reg_corr_mat = np.empty((all_activity.shape[0], n_regs), dtype=np.float32)
    for i in range(all_activity.shape[0]):
        for j in range(n_regs):
            reg_corr_mat[i, j] = np.corrcoef(all_activity[i, :], reg_trans[:, j])[0, 1]

    reg_corr_th = 0.6
    reg_corr_mat[np.abs(reg_corr_mat) < reg_corr_th] = 0
    # exclude cells where correlations with either sensory regressor are NaN *but not* cells for which one of the
    # motor-regressors might be NaN (as we don't expect all motor types to be executed in each plane)
    no_nan = np.sum(np.isnan(reg_corr_mat[:, :n_regs + 1]), 1) == 0
    reg_corr_mat = reg_corr_mat[no_nan, :]

    ab_thresh = np.sum(reg_corr_mat > 0, 1) > 0

    # plot regressor correlation matrix - all units no clustering
    fig, ax = pl.subplots()
    sns.heatmap(reg_corr_mat[np.argsort(ab_thresh)[::-1], :], vmin=-1, vmax=1, center=0, yticklabels=75000)
    ax.set_title('Regressor correlations, thresholded at ' + str(reg_corr_th))

    # remove all rows that don't have at least one above-threshold correlation
    # NOTE: The copy statement below is required to prevent a stale copy of the full-sized array to remain in memory
    reg_corr_mat = reg_corr_mat[ab_thresh, :].copy()

    km = max_cluster(np.nanargmax(reg_corr_mat, 1))
    # km.fit(reg_corr_mat)
    # plot sorted by cluster identity
    fig, ax = pl.subplots()
    sns.heatmap(reg_corr_mat[np.argsort(km.labels_), :], vmin=-1, vmax=1, center=0, yticklabels=5000)
    # plot cluster boundaries
    covered = 0
    for i in range(km.n_clusters):
        covered += np.sum(km.labels_ == i)
        ax.plot([0, n_regs+1], [km.labels_.size-covered, km.labels_.size-covered], 'k')
    ax.set_title('Above threshold correlations clustered and sorted')

    membership = np.zeros(exp_id.size, dtype=np.float32) - 1
    temp_no_nan_aa = np.zeros(np.sum(no_nan_aa), dtype=np.float32) - 1
    temp_no_nan = np.zeros(np.sum(no_nan), dtype=np.float32) - 1
    temp_no_nan[ab_thresh] = km.labels_
    temp_no_nan_aa[no_nan] = temp_no_nan
    membership[no_nan_aa] = temp_no_nan_aa
    assert membership.size == no_nan_aa.size
    assert membership[no_nan_aa].size == all_activity.shape[0]
    assert membership.size == sum([len(e.graph_info) for e in exp_data])
    mship_nonan = membership[no_nan_aa]

    cells_on = all_activity[np.logical_and(mship_nonan > -1, mship_nonan < 3), :]
    t_avg_on = trial_average(cells_on, False, 2)
    cells_off = all_activity[np.logical_and(mship_nonan > 2, mship_nonan < 6), :]
    t_avg_off = trial_average(cells_off, False, 2)
    act_to_plot = np.vstack((dff(t_avg_on), dff(t_avg_off)))
    fig, ax = pl.subplots()
    sns.heatmap(act_to_plot, xticklabels=150, yticklabels=250, vmin=-3, vmax=3, ax=ax, rasterized=True)
