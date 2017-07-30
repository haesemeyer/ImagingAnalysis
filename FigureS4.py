# Script to aggregate plots for Supplemental Figure 4 of Heat-Imaging paper
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import RegionContainer, assign_region_label, SLHRepeatExperiment, MotorContainer, IndexingMatrix
from multiExpAnalysis import get_stack_types, dff
from typing import List
import matplotlib as mpl
import pandas
import sys
from analyzeSensMotor import RegionResults
from motorPredicates import left_bias_bouts, right_bias_bouts

sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')
from mhba_basic import Crosscorrelation


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

if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS4/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    membership = np.array(dfile['membership'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    mship_nonan = membership[no_nan_aa]
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    exp_id = np.array(dfile['exp_id'])
    eid_nonan = exp_id[no_nan_aa]
    # limit sourceFiles to the contents of all_activity
    sourceFiles = [(g[0], e.original_time_per_frame) for e in exp_data for g in e.graph_info]
    sourceFiles = [sf for i, sf in enumerate(sourceFiles) if no_nan_aa[i]]
    tf_centroids = np.array(dfile['tf_centroids'])[no_nan_aa, :]
    all_activity = np.array(dfile["all_activity"])
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

    stack_types = get_stack_types(exp_data)[no_nan_aa]
    # get indices of on-type cells in regions (since everything is limited to no_nan_aa these should match)
    all_rl = build_all_region_labels()
    # load motor related data
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/sh_motor_system.hdf5", "r")
    membership_motor = np.array(motor_store["membership_motor"])[no_nan_aa]
    motor_corrs = np.array(motor_store["motor_corrs"])
    cluster_labels = np.array(motor_store["cluster_labels"])
    motor_cells = (cluster_labels > -1)
    motor_store.close()

    # plot motor clustering
    labels = ["All motor", "Flicks", "Right flicks", "Left flicks", "Swims", "Stim only", "No stim"]
    motor_corrs_sig = motor_corrs[motor_cells, :].copy()
    plot_corrs = []
    for c in np.unique(cluster_labels[motor_cells]):
        plot_corrs.append(motor_corrs_sig[cluster_labels[motor_cells] == c, :])
        # append "spacer" - NOTE: This means that y-axis becomes meaningless
        plot_corrs.append(np.full((25, motor_corrs_sig.shape[1]), np.nan))
    fig, ax = pl.subplots()
    sns.heatmap(np.vstack(plot_corrs), vmin=0, vmax=1, yticklabels=False,
                xticklabels=labels, center=0, rasterized=True)
    fig.savefig(save_folder + "sh_Motor_correlations.pdf", type="pdf", dpi=150)

    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    itime = np.linspace(0, all_activity.shape[1] / 5, all_activity.shape[1] + 1)
    mc_all_raw = MotorContainer(sourceFiles, itime, 0, hdf5_store=tailstore)
    mc_fleft_raw = MotorContainer(sourceFiles, itime, 0, tdd=mc_all_raw.tdd, predicate=left_bias_bouts)
    mc_fright_raw = MotorContainer(sourceFiles, itime, 0, tdd=mc_all_raw.tdd, predicate=right_bias_bouts)

    # compute and plot motor cross-correlations for on-motor and off-motor categories
    max_lag = 50
    # compute crosscorrelations for all_motor, on_motor and off_motor against bouts while stimulus is on and off
    cc_all_stim = []
    cc_all_nostim = []
    cc_onmot_stim = []
    cc_onmot_nostim = []
    cc_offmot_stim = []
    cc_offmot_nostim = []
    cc_on_stim = []
    cc_on_nostim = []
    cc_off_stim = []
    cc_off_nostim = []
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    poa_data = pickle.loads(np.array(storage["POA"]))  # type: RegionResults
    poa_sens = np.mean(poa_data.region_acts[poa_data.region_mem == 1, :], 0)
    storage.close()
    # we create the following stimulus-motor regressors:
    # A multiplication of mc_all with a 0/1 thresholded version of the activity trace (only events when active)
    poa_thresh = 0.25  # POA slow ON cells are considered active above this threshold, based on looking at average
    poa_on = poa_sens > 0.25
    poa_off = poa_sens <= 0.25

    for i in range(all_activity.shape[0]):
        on_starts = mc_all_raw[i] * poa_on
        off_starts = mc_all_raw[i] * poa_off
        # only consider planes where we have both on and off starts
        if np.sum(on_starts) == 0 or np.sum(off_starts) == 0:
            continue
        if membership_motor[i] == 0:
            cc_all_stim.append(Crosscorrelation(all_activity[i, :], on_starts, max_lag))
            cc_all_nostim.append(Crosscorrelation(all_activity[i, :], off_starts, max_lag))
        elif membership_motor[i] == 5:
            cc_onmot_stim.append(Crosscorrelation(all_activity[i, :], on_starts, max_lag))
            cc_onmot_nostim.append(Crosscorrelation(all_activity[i, :], off_starts, max_lag))
        elif membership_motor[i] == 6:
            cc_offmot_stim.append(Crosscorrelation(all_activity[i, :], on_starts, max_lag))
            cc_offmot_nostim.append(Crosscorrelation(all_activity[i, :], off_starts, max_lag))

    # plot cross-correlations
    fig, (ax_on, ax_off) = pl.subplots(ncols=2, sharey=True)
    t = np.arange(-1*max_lag, max_lag+1) / 5
    sns.tsplot(data=cc_all_stim, time=t, ci=99.9, ax=ax_on, color='C1')
    sns.tsplot(data=cc_onmot_stim, time=t, ci=99.9, ax=ax_on, color='C3')
    sns.tsplot(data=cc_offmot_stim, time=t, ci=99.9, ax=ax_on, color='C0')
    ax_on.set_xlabel('Time lag around stim bout [s]')
    ax_on.set_ylabel('Cross correlation')
    sns.tsplot(data=cc_all_nostim, time=t, ci=99.9, ax=ax_off, color='C1')
    sns.tsplot(data=cc_onmot_nostim, time=t, ci=99.9, ax=ax_off, color='C3')
    sns.tsplot(data=cc_offmot_nostim, time=t, ci=99.9, ax=ax_off, color='C0')
    ax_off.set_xlabel('Time lag around no-stim bout [s]')
    ax_off.set_ylabel('Cross correlation')
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "sh_onoffmotor_MTA.pdf", type="pdf")

    # compute and plot motor-triggered averages for flick-left and flick-right categories
    cc_all_left = []
    cc_all_right = []
    cc_left_left = []
    cc_left_right = []
    cc_right_left = []
    cc_right_right = []
    for i in range(all_activity.shape[0]):
        left = mc_fleft_raw[i]
        right = mc_fright_raw[i]
        if left.sum() == 0 or right.sum() == 0:
            continue
        if membership_motor[i] == 2:  # note: predicate names are actually side-inverted!
            # right flick cell
            cc_right_left.append(Crosscorrelation(all_activity[i, :], left, max_lag))
            cc_right_right.append(Crosscorrelation(all_activity[i, :], right, max_lag))
        elif membership_motor[i] == 3:
            # left flick cell
            cc_left_left.append(Crosscorrelation(all_activity[i, :], left, max_lag))
            cc_left_right.append(Crosscorrelation(all_activity[i, :], right, max_lag))
        elif membership_motor[i] == 0:
            # all motor cell
            cc_all_left.append(Crosscorrelation(all_activity[i, :], left, max_lag))
            cc_all_right.append(Crosscorrelation(all_activity[i, :], right, max_lag))
            # plot cross-correlations
    fig, (ax_left, ax_right) = pl.subplots(ncols=2, sharey=True)
    t = np.arange(-1 * max_lag, max_lag + 1) / 5
    sns.tsplot(data=cc_all_left, time=t, ci=99.9, ax=ax_left, color='C1')
    sns.tsplot(data=cc_left_left, time=t, ci=99.9, ax=ax_left, color='C4')
    sns.tsplot(data=cc_right_left, time=t, ci=99.9, ax=ax_left, color='C2')
    ax_left.set_xlabel('Time lag around right flick [s]')
    ax_left.set_ylabel('Cross correlation')
    sns.tsplot(data=cc_all_right, time=t, ci=99.9, ax=ax_right, color='C1')
    sns.tsplot(data=cc_left_right, time=t, ci=99.9, ax=ax_right, color='C4')
    sns.tsplot(data=cc_right_right, time=t, ci=99.9, ax=ax_right, color='C2')
    ax_right.set_xlabel('Time lag around left flick [s]')
    ax_right.set_ylabel('Cross correlation')
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "sh_leftright_MTA.pdf", type="pdf")

    # load original membership_motor for anatomy and shuffle it
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "r")
    membership_motor = np.array(motor_store["membership_motor"])[no_nan_aa]
    motor_store.close()
    np.random.shuffle(membership_motor)

    # plot whole-brain overview of motor related cells
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "MAIN"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 250, False)
    fig, ax = pl.subplots()
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.1, c='k')
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor > -1), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor > -1), 1], s=1, alpha=0.4, c="C1")
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_Motor_BrainTop.pdf", type="pdf")

    fig, ax = pl.subplots()
    half_brain = np.logical_and(tf_centroids[:, 0] < np.nanmedian(tf_centroids[stack_types == "MAIN", 0]),
                                stack_types == "MAIN")
    ax.scatter(tf_centroids[ix_bground, 1], tf_centroids[ix_bground, 2], s=1, alpha=0.1, c='k')
    ax.scatter(tf_centroids[np.logical_and(half_brain, membership_motor > -1), 1],
               tf_centroids[np.logical_and(half_brain, membership_motor > -1), 2], s=1, alpha=0.4, c="C1")
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_Motor_BrainSide.pdf", type="pdf")

    def part_of(regions):
        if type(regions) == str:
            return all_rl == regions
        else:
            p_of = np.zeros(all_rl.shape[0], bool)
            for r in regions:
                p_of = np.logical_or(p_of, all_rl == r)
        return p_of

    r_to_count = [
        ["A_HB_L", "A_HB_R"],
        ["Cerebellum_L", "Cerebellum_R"],
        "ncMLF",
        ["Tect_L", "Tect_R"],
        ["Hab_L", "Hab_R"],
        ["D_FB_L", "D_FB_R"],
        "SubPallium"
    ]
    labels = ["Ant. Hindbrain", "Cerebellum", "ncMLF", "Tectum", "Habenula", "Pallium", "Subpallium"]

    mot_fracs = {labels[i]: [np.sum(np.logical_and(membership_motor > -1, part_of(r))) / np.sum(part_of(r)) * 100]
                 for i, r in enumerate(r_to_count)}

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(mot_fracs), ax=ax, order=labels)
    ax.set_ylim(0, 6)
    ax.set_ylabel("Fraction of motor related cells")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_PerRegion_Motor_Fraction.pdf", type="pdf")
