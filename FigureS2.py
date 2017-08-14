# Script to aggregate plots for  Supplemental Figure 2 of Heat-Imaging paper
# A-B) Top and side view of ON/OFF cell distribution after spatial shuffle
# C) Barplot of fraction of responsive cells in same regions as Figure2
# D) Distance comparison heat-sens vs. other
# E) Distance comparison ON-ON vs ON-OFF
# F) Distance comparison OFF-OFF vs OFF-ON

import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib as mpl
import h5py
import numpy as np
from typing import List
from mh_2P import SLHRepeatExperiment, RegionContainer, assign_region_label
from multiExpAnalysis import get_stack_types, min_dist
import pickle
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


if __name__ == '__main__':
    save_folder = "./HeatImaging/FigureS2/"
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

    def partition(a, nchunks=10):
        """
        Partion an array in n nearly equal sized chunks
        """
        chunks = []
        chnksize = a.size // nchunks
        for i in range(nchunks):
            s = i*chnksize
            if i == nchunks-1:
                e = a.size
            else:
                e = (i+1) * chnksize
            chunks.append(a[s:e])
        return chunks

    stack_types = get_stack_types(exp_data)[no_nan_aa]
    all_rl = build_all_region_labels()
    # get shuffled ON and OFF cell labels
    np.random.shuffle(mship_nonan)
    on_cells = np.logical_and(mship_nonan > -1, mship_nonan < 3)
    on_indices = np.arange(tf_centroids.shape[0])[np.logical_and(stack_types == "MAIN", on_cells)]
    np.random.shuffle(on_indices)
    on_indices = partition(on_indices)
    off_cells = np.logical_and(mship_nonan > 3, mship_nonan < 6)
    off_indices = np.arange(tf_centroids.shape[0])[np.logical_and(stack_types == "MAIN", off_cells)]
    np.random.shuffle(off_indices)
    off_indices = partition(off_indices)

    # plot whole-brain overview of ON and OFF types
    fig, ax = pl.subplots()
    for i in range(10):
        ax.scatter(tf_centroids[on_indices[i], 0],
                   tf_centroids[on_indices[i], 1], s=1, alpha=0.3, c='m')
        ax.scatter(tf_centroids[off_indices[i], 0],
                   tf_centroids[off_indices[i], 1], s=1, alpha=0.3, c="g")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_ON_OFF_BrainTop.pdf", type="pdf")

    fig, ax = pl.subplots()
    half_brain = np.logical_and(tf_centroids[:, 0] < np.nanmedian(tf_centroids[stack_types == "MAIN", 0]),
                                stack_types == "MAIN")
    on_indices = np.arange(tf_centroids.shape[0])[np.logical_and(half_brain, on_cells)]
    np.random.shuffle(on_indices)
    on_indices = partition(on_indices)
    off_cells = np.logical_and(mship_nonan > 3, mship_nonan < 6)
    off_indices = np.arange(tf_centroids.shape[0])[np.logical_and(half_brain, off_cells)]
    np.random.shuffle(off_indices)
    off_indices = partition(off_indices)
    for i in range(10):
        ax.scatter(tf_centroids[on_indices[i], 1],
                   tf_centroids[on_indices[i], 2], s=1, alpha=0.3, c='m')
        ax.scatter(tf_centroids[off_indices[i], 1],
                   tf_centroids[off_indices[i], 2], s=1, alpha=0.3, c="g")
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_ON_OFF_BrainSide.pdf", type="pdf")

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


    on_fracs = {labels[i]: [np.sum(np.logical_and(on_cells, part_of(r))) / np.sum(part_of(r)) * 100] for i, r in
                enumerate(r_to_count)}
    off_fracs = {labels[i]: [np.sum(np.logical_and(off_cells, part_of(r))) / np.sum(part_of(r)) * 100] for i, r in
                 enumerate(r_to_count)}

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(on_fracs), ax=ax, order=labels)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Fraction of ON cells")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_PerRegion_ON_Fraction.pdf", type="pdf")

    fig, ax = pl.subplots()
    sns.barplot(data=pandas.DataFrame(off_fracs), ax=ax, order=labels)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Fraction of OFF cells")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_PerRegion_OFF_Fraction.pdf", type="pdf")

    # plot ON-ON and ON-OFF distances
    mship_main = mship_nonan[stack_types == "MAIN"]
    cents_main = tf_centroids[stack_types == "MAIN", :]
    on = np.logical_and(mship_main > -1, mship_main < 4)
    off = np.logical_and(mship_main > 3, mship_main < 6)
    c_on = cents_main[on, :]
    c_off = cents_main[off, :]
    subs_c_on = c_on[np.random.choice(np.arange(c_on.shape[0]), c_off.shape[0], False), :]
    d_on_on = min_dist(subs_c_on, subs_c_on, avgSmallest=2)
    d_on_off = min_dist(subs_c_on, c_off, avgSmallest=2)
    d_off_off = min_dist(c_off, c_off, avgSmallest=2)
    d_off_on = min_dist(c_off, subs_c_on, avgSmallest=2)

    fig, ax = pl.subplots()
    sns.kdeplot(d_on_on, cut=0, ax=ax, color="C0")
    sns.kdeplot(d_on_off, cut=0, ax=ax, color="C1")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Distance [um]")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_ON-ON_ON-OFF_DistanceCompare.pdf", type="pdf")

    fig, ax = pl.subplots()
    sns.kdeplot(d_off_off, cut=0, ax=ax, color="C0")
    sns.kdeplot(d_off_on, cut=0, ax=ax, color="C1")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Distance [um]")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_OFF-OFF_OFF-ON_DistanceCompare.pdf", type="pdf")

    # plot stim-stim vs. stim non-stim distances
    stim_act = np.logical_and(mship_main > -1, mship_main < 6)
    not_stim = mship_main == -1
    c_s_a = cents_main[stim_act, :]
    c_n_s = cents_main[not_stim, :]
    subs_c_n_s = c_n_s[np.random.choice(np.arange(c_n_s.shape[0]), c_s_a.shape[0], False), :]
    d_s_s = min_dist(c_s_a, c_s_a, avgSmallest=2)
    d_s_ns = min_dist(c_s_a, subs_c_n_s, avgSmallest=2)
    d_ns_ns = min_dist(subs_c_n_s, subs_c_n_s, avgSmallest=2)
    d_ns_s = min_dist(subs_c_n_s, c_s_a, avgSmallest=2)

    fig, ax = pl.subplots()
    sns.kdeplot(d_s_s, cut=0, ax=ax, color="C0")
    sns.kdeplot(d_s_ns, cut=0, ax=ax, color="C1")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Distance [um]")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "sh_stim-stim_stim-nostim_DistanceCompare.pdf", type="pdf")
