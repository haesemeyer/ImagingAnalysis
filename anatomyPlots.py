# Script to perform anatomical plots of cell locations
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


def PlotRegionHeatFraction_TopDown(rlabels, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    pal = sns.color_palette("plasma", n_colors=15)
    in_region = np.zeros(all_rl.shape[0], dtype=bool)
    if type(rlabels) is str:
        rlabels = [rlabels]
    for r in rlabels:
        in_region = np.logical_or(in_region, all_rl == r)
    total = in_region.sum()
    heat = np.logical_and(in_region, on_cells).sum() + np.logical_and(in_region, off_cells).sum()
    fraction = heat / total
    col_index = int((fraction * 15) / 0.15)
    if col_index > 14:
        col_index = 14
    # subsample
    ix_to_plot = np.arange(in_region.sum())
    ix_to_plot = np.random.choice(ix_to_plot, ix_to_plot.size // 20)
    ax.scatter(tf_centroids[in_region, 0][ix_to_plot], tf_centroids[in_region, 1][ix_to_plot], s=1, c=pal[col_index]
               , alpha=0.3)
    return fraction, heat, total


def PlotRegionHeatFraction_Side(rlabels, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    pal = sns.color_palette("plasma", n_colors=15)
    in_region = np.zeros(all_rl.shape[0], dtype=bool)
    if type(rlabels) is str:
        rlabels = [rlabels]
    for r in rlabels:
        in_region = np.logical_or(in_region, all_rl == r)
    total = in_region.sum()
    heat = np.logical_and(in_region, on_cells).sum() + np.logical_and(in_region, off_cells).sum()
    fraction = heat / total
    col_index = int((fraction * 15) / 0.15)
    if col_index > 14:
        col_index = 14
    # subsample
    ix_to_plot = np.arange(in_region.sum())
    ix_to_plot = np.random.choice(ix_to_plot, ix_to_plot.size // 20)
    ax.scatter(tf_centroids[in_region, 1][ix_to_plot], tf_centroids[in_region, 2][ix_to_plot], s=1, c=pal[col_index],
               alpha=0.3)
    return fraction, heat, total


if __name__ == "__main__":
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
    on_cells = np.logical_and(mship_nonan > -1, mship_nonan < 3)
    off_cells = np.logical_and(mship_nonan > 3, mship_nonan < 6)

    # plot whole-brain overview of ON and OFF types
    fig, ax = pl.subplots()
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", on_cells), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", on_cells), 1], s=1, alpha=0.3, c='m',
               label="ON cells")
    ax.scatter(tf_centroids[np.logical_and(stack_types == "MAIN", off_cells), 0],
               tf_centroids[np.logical_and(stack_types == "MAIN", off_cells), 1], s=1, alpha=0.3, c="g",
               label="OFF cells")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    half_brain = np.logical_and(tf_centroids[:, 0] < np.nanmedian(tf_centroids[stack_types == "MAIN", 0]),
                                stack_types == "MAIN")
    ax.scatter(tf_centroids[np.logical_and(half_brain, on_cells), 1],
               tf_centroids[np.logical_and(half_brain, on_cells), 2], s=1, alpha=0.3, c='m',
               label="ON cells")
    ax.scatter(tf_centroids[np.logical_and(half_brain, off_cells), 1],
               tf_centroids[np.logical_and(half_brain, off_cells), 2], s=1, alpha=0.3, c="g",
               label="OFF cells")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)

    # plot both trigeminal ganglia top-down
    fig, ax = pl.subplots()
    in_tleft = all_rl == "TG_L"
    hull = ConvexHull(tf_centroids[in_tleft, :2])
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "TG_LEFT"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 2, False)
    tleft_cells = np.logical_and(tf_centroids[:, 2] >= np.min(tf_centroids[all_rl == "TG_L", 2]),
                                 tf_centroids[:, 2] <= np.max(tf_centroids[all_rl == "TG_L", 2]))
    tleft_cells = np.logical_and(tleft_cells, stack_types == "TG_LEFT")
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.2, c='k', label="")
    ax.scatter(tf_centroids[np.logical_and(stack_types == "TG_LEFT", on_cells), 0],
               tf_centroids[np.logical_and(stack_types == "TG_LEFT", on_cells), 1], s=3, alpha=0.3, c='m',
               label="ON cells")
    ax.scatter(tf_centroids[np.logical_and(stack_types == "TG_LEFT", off_cells), 0],
               tf_centroids[np.logical_and(stack_types == "TG_LEFT", off_cells), 1], s=3, alpha=0.3, c="g",
               label="OFF cells")
    for simplex in hull.simplices:
        ax.plot(tf_centroids[in_tleft, :][simplex, 0], tf_centroids[in_tleft, :][simplex, 1], 'C0')
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    in_tright = all_rl == "TG_R"
    hull = ConvexHull(tf_centroids[in_tright, :2])
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "TG_RIGHT"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 2, False)
    tright_cells = np.logical_and(tf_centroids[:, 2] >= np.min(tf_centroids[all_rl == "TG_R", 2]),
                                  tf_centroids[:, 2] <= np.max(tf_centroids[all_rl == "TG_R", 2]))
    tright_cells = np.logical_and(tright_cells, stack_types == "TG_RIGHT")
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.2, c='k', label="")
    ax.scatter(tf_centroids[np.logical_and(tright_cells, on_cells), 0],
               tf_centroids[np.logical_and(tright_cells, on_cells), 1], s=3, alpha=0.3, c='m',
               label="ON cells")
    ax.scatter(tf_centroids[np.logical_and(tright_cells, off_cells), 0],
               tf_centroids[np.logical_and(tright_cells, off_cells), 1], s=3, alpha=0.3, c="g",
               label="OFF cells")
    for simplex in hull.simplices:
        ax.plot(tf_centroids[in_tright, :][simplex, 0], tf_centroids[in_tright, :][simplex, 1], 'C0')
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)

    # plot map of heat-responsive fraction for different example regions - NOTE: ordered from ventral to dorsal!
    r_to_plot = ["SubPallium", "D_FB_L", "D_FB_R", "Hab_L", "Hab_R", "Tect_L", "Tect_R", "Cerebellum_L",
                 "Cerebellum_R", "Rh6_L", "Rh6_R"]
    fig, ax = pl.subplots()
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "MAIN"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 100, False)
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.2, c='k')
    for r in r_to_plot:
        PlotRegionHeatFraction_TopDown(r, ax)
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    r_side = ["PreOptic", "SubPallium", "D_FB_R", "Hab_R", "Tect_R", "Cerebellum_R", "Rh6_R"]
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "MAIN"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 100, False)
    ax.scatter(tf_centroids[ix_bground, 1], tf_centroids[ix_bground, 2], s=1, alpha=0.2, c='k')
    for r in r_side:
        PlotRegionHeatFraction_Side(r, ax)
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "TG_LEFT"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 2, False)
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.2, c='k')
    for r in r_to_plot:
        PlotRegionHeatFraction_TopDown("TG_L", ax)
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    ix_bground = np.arange(tf_centroids.shape[0])[stack_types == "TG_RIGHT"]
    ix_bground = np.random.choice(ix_bground, ix_bground.size // 2, False)
    ax.scatter(tf_centroids[ix_bground, 0], tf_centroids[ix_bground, 1], s=1, alpha=0.2, c='k')
    for r in r_to_plot:
        PlotRegionHeatFraction_TopDown("TG_R", ax)
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)

    # for each region get the on and off fraction and plot numbers
    r_to_count = [
        ["TG_L", "TG_R"],
        ["Rh6_L", "Rh6_R"],
        ["Cerebellum_L", "Cerebellum_R"],
        ["Tect_L", "Tect_R"],
        ["Hab_L", "Hab_R"],
        ["D_FB_L", "D_FB_R"],
        "SubPallium",
        "PreOptic"
    ]
    labels = ["Trigeminal", "Rh6", "Cerebellum", "Tectum", "Habenula", "Pallium", "Subpallium", "POA"]

    def part_of(regions):
        if type(regions) == str:
            return all_rl == regions
        else:
            p_of = np.zeros(all_rl.shape[0], bool)
            for r in regions:
                p_of = np.logical_or(p_of, all_rl == r)
        return p_of

    on_fracs = {labels[i]: [np.sum(np.logical_and(on_cells, part_of(r))) / np.sum(part_of(r)) * 100] for i, r in enumerate(r_to_count)}
    off_fracs = {labels[i]: [np.sum(np.logical_and(off_cells, part_of(r))) / np.sum(part_of(r)) * 100] for i, r in enumerate(r_to_count)}

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(on_fracs), ax=ax, order=labels)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Fraction of ON cells")
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(off_fracs), ax=ax, order=labels)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Fraction of OFF cells")
    sns.despine(fig, ax)
