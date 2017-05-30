# Script to aggregate plots for Figure 4 of Heat-Imaging paper
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import RegionContainer, assign_region_label, SLHRepeatExperiment
from multiExpAnalysis import get_stack_types
from typing import List
import matplotlib as mpl
from scipy.spatial import ConvexHull
import pandas


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


def PlotRegionMotorFraction_TopDown(rlabels, ax=None, plot_color_bar=False):
    pal = sns.color_palette("plasma", n_colors=10)
    max_fraction = 0.05
    if plot_color_bar:
        sns.palplot(pal, 0.5)
        cb_labels = np.linspace(0, max_fraction * 100, 10, endpoint=False)
        pl.xticks(np.arange(10), [str(a) for a in cb_labels])
        return None
    if ax is None:
        fig, ax = pl.subplots()
    in_region = np.zeros(all_rl.shape[0], dtype=bool)
    if type(rlabels) is str:
        rlabels = [rlabels]
    for r in rlabels:
        in_region = np.logical_or(in_region, all_rl == r)
    total = in_region.sum()
    motor = np.logical_and(in_region, membership_motor > -1).sum()
    fraction = motor / total
    col_index = int((fraction * 10) / max_fraction)
    if col_index > 9:
        col_index = 9
    # subsample
    ix_to_plot = np.arange(in_region.sum())
    ix_to_plot = np.random.choice(ix_to_plot, ix_to_plot.size // 20)
    ax.scatter(tf_centroids[in_region, 0][ix_to_plot], tf_centroids[in_region, 1][ix_to_plot], s=1, c=pal[col_index]
               , alpha=0.3)
    return fraction, motor, total


def PlotRegionMotorFraction_Side(rlabels, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    max_fraction = 0.05
    pal = sns.color_palette("plasma", n_colors=10)
    in_region = np.zeros(all_rl.shape[0], dtype=bool)
    if type(rlabels) is str:
        rlabels = [rlabels]
    for r in rlabels:
        in_region = np.logical_or(in_region, all_rl == r)
    total = in_region.sum()
    motor = np.logical_and(in_region, membership_motor > -1).sum()
    fraction = motor / total
    col_index = int((fraction * 10) / max_fraction)
    if col_index > 9:
        col_index = 9
    # subsample
    ix_to_plot = np.arange(in_region.sum())
    ix_to_plot = np.random.choice(ix_to_plot, ix_to_plot.size // 20)
    ax.scatter(tf_centroids[in_region, 1][ix_to_plot], tf_centroids[in_region, 2][ix_to_plot], s=1, c=pal[col_index],
               alpha=0.3)
    return fraction, motor, total


if __name__ == "__main__":
    save_folder = "./HeatImaging/Figure4/"
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
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "r")
    membership_motor = np.array(motor_store["membership_motor"])[no_nan_aa]
    motor_corrs = np.array(motor_store["motor_corrs"])
    cluster_labels = np.array(motor_store["cluster_labels"])
    motor_cells = (cluster_labels > -1)
    motor_store.close()

    # plot motor clustering
    labels = ["All motor", "Flicks", "Right flicks", "Left flicks", "Swims", "Stim only", "No stim"]
    motor_corrs_sig = motor_corrs[motor_cells, :].copy()
    fig, ax = pl.subplots()
    sns.heatmap(motor_corrs_sig[np.argsort(cluster_labels[motor_cells]), :], vmin=0, vmax=1, yticklabels=2500,
                xticklabels=labels, cmap="viridis", rasterized=True)
    # plot cluster boundaries
    covered = 0
    for i in range(motor_corrs_sig.shape[1]):
        covered += np.sum(cluster_labels[motor_cells] == i)
        ax.plot([0, motor_corrs_sig.shape[1] + 1], [cluster_labels[motor_cells].size-covered,
                                                    cluster_labels[motor_cells].size-covered], 'k')
    fig.savefig(save_folder + "Motor_correlations.pdf", type="pdf")

    # plot whole-brain overview of motor related cells
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "MAIN"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 250, False)
    fig, ax = pl.subplots()
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.1, c='k')
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor > -1), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor > -1), 1], s=1, alpha=0.4, c="C1")
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Motor_BrainTop.pdf", type="pdf")

    fig, ax = pl.subplots()
    half_brain = np.logical_and(tf_centroids[:, 0] < np.nanmedian(tf_centroids[stack_types == "MAIN", 0]),
                                stack_types == "MAIN")
    ax.scatter(tf_centroids[ix_bground, 1], tf_centroids[ix_bground, 2], s=1, alpha=0.1, c='k')
    ax.scatter(tf_centroids[np.logical_and(half_brain, membership_motor > -1), 1],
               tf_centroids[np.logical_and(half_brain, membership_motor > -1), 2], s=1, alpha=0.4, c="C1")
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Motor_BrainSide.pdf", type="pdf")

    # Plot left-flick vs. right-flick cells
    fig, ax = pl.subplots()
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.1, c='k')
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 2), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 2), 1], s=2, c="C2",
               label="Flick right")
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 3), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 3), 1], s=2, c="C4",
               label="Flick left")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Motor_LFick_RFlick.pdf", type="pdf")

    # Plot stim_only vs. no_stim cells
    fig, ax = pl.subplots()
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.1, c='k')
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 5), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 5), 1], s=2, c="C3",
               label="stim only")
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 6), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", membership_motor == 6), 1], s=2, c="C0",
               label="no stim")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Motor_Stim_NoStim.pdf", type="pdf")

    # plot map of motor-responsive fraction for different example regions - NOTE: ordered from ventral to dorsal!
    r_to_plot = ["D_FB_L", "D_FB_R", "Hab_L", "Hab_R", "ncMLF", "Tect_L", "Tect_R", "A_HB_L", "A_HB_R",
                 "Cerebellum_L", "Cerebellum_R"]
    fig, ax = pl.subplots()
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "MAIN"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 100, False)
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.2, c='k')
    for r in r_to_plot:
        PlotRegionMotorFraction_TopDown(r, ax)
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "MotorFractionMap_Brain_Top.pdf", type="pdf")

    fig, ax = pl.subplots()
    r_side = ["D_FB_R", "Hab_R", "ncMLF", "Tect_R", "A_HB_R", "Cerebellum_R"]
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "MAIN"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 100, False)
    ax.scatter(tf_centroids[ix_bground, 1], tf_centroids[ix_bground, 2], s=1, alpha=0.2, c='k')
    for r in r_side:
        PlotRegionMotorFraction_Side(r, ax)
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "MotorFractionMap_Brain_Side.pdf", type="pdf")

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
    fig.savefig(save_folder + "PerRegion_Motor_Fraction.pdf", type="pdf")
