# Script to aggregate plots for  Supplemental Figure 1 of Heat-Imaging paper
# A) Image stabilization pipeline
# B) Stabilizer movements (either RFP or gcamp experiments??)
# C) Effect of image stabilization on stimulus correlations in RFP stacks
# D) Tail angle examples of swims and flicks

import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib as mpl
import h5py
import numpy as np
from motorPredicates import bias
from mh_2P import MotorContainer, SLHRepeatExperiment
import pickle
from typing import List
from scipy.stats import mode
import os

if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS1/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    no_nan_aa = np.array(dfile['no_nan_aa'])
    nframes = np.array(dfile['all_activity']).shape[1]
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    # limit sourceFiles to the contents of all_activity
    sourceFiles = [(g[0], e.original_time_per_frame) for e in exp_data for g in e.graph_info]
    sourceFiles = [sf for i, sf in enumerate(sourceFiles) if no_nan_aa[i]]
    dfile.close()
    # create motor containers
    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    itime = np.linspace(0, nframes / 5, nframes + 1)
    mc_all_raw = MotorContainer(sourceFiles, itime, 0, hdf5_store=tailstore)

    # plot z-stabilization effect
    stab_file = h5py.File("E:/Dropbox/2P_Data/RFP_Boxcar_StabBench/stable_test.hdf5", 'r')
    corr_on_nostable = np.array(stab_file["corr_on_nostable"])
    corr_on_stable = np.array(stab_file["corr_on_stable"])
    corr_on_shuffle = np.array(stab_file["corr_on_shuffle"])
    stab_file.close()
    h_stable, bins = np.histogram(corr_on_stable, np.linspace(-1, 1))
    b_centers = bins[:-1] + np.diff(bins)/2
    h_stable = h_stable / h_stable.sum()
    h_nostable = np.histogram(corr_on_nostable, bins)[0]
    h_nostable = h_nostable / h_nostable.sum()
    h_shuffle = np.histogram(corr_on_shuffle, bins)[0]
    h_shuffle = h_shuffle / h_shuffle.sum()
    fig, ax = pl.subplots()
    ax.plot(b_centers, h_nostable, label="Not stabilized")
    ax.plot(b_centers, h_stable, label="Stabilized")
    ax.plot(b_centers, h_shuffle, label="Shuffle")
    ax.set_xlabel("Stimulus correlation")
    ax.set_ylabel("Proportion")
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Stabilization_Correlation.pdf", type="pdf")

    # plot example tail-traces of bouts with different biases
    def plot_bout(ix, ax):
        fname, index = lookup_table["filename"][ix], lookup_table["index"][ix]
        td = mc_all_raw.tdd[fname]
        b = td.bouts[index, :].astype(int)
        ca = td.cumAngles[b[0]-20:b[1]+21]
        t = (np.arange(ca.size)-20) / 100
        ax.plot(t, ca)

    biases = []
    lookup_table = {"filename": [], "index": []}
    # make sure mc_all_raw's dictionary is set up
    mc_all_raw.avg_motor_output
    for k in mc_all_raw.tdd.fileNames:
        td = mc_all_raw.tdd[k]
        mca = mode(td.cumAngles)[0]
        if td.bouts is not None:
            for i, b in enumerate(td.bouts.astype(int)):
                biases.append(bias(b[0], b[1]+1, td.cumAngles, mca))
                lookup_table["filename"].append(k)
                lookup_table["index"].append(i)
    biases = np.array(biases)
    lfl = np.nonzero(biases > 0.9)[0]
    rfl = np.nonzero(biases < -0.9)[0]
    sw_l = np.nonzero(np.logical_and(biases < 0.5, biases > 0.3))[0]
    sw_r = np.nonzero(np.logical_and(biases > -0.5, biases < -0.3))[0]
    mid = np.nonzero(np.logical_and(biases > -0.1, biases < 0.1))[0]
    fig, axes = pl.subplots(nrows=3, ncols=3, sharey=True, sharex=True)
    plot_bout(np.random.choice(lfl), axes[0, 0])
    plot_bout(np.random.choice(lfl), axes[1, 0])
    plot_bout(np.random.choice(lfl), axes[2, 0])
    # plot_bout(np.random.choice(sw_l), axes[0, 1])
    # plot_bout(np.random.choice(sw_l), axes[1, 1])
    # plot_bout(np.random.choice(sw_l), axes[2, 1])
    plot_bout(np.random.choice(mid), axes[0, 1])
    plot_bout(np.random.choice(mid), axes[1, 1])
    plot_bout(np.random.choice(mid), axes[2, 1])
    # plot_bout(np.random.choice(sw_r), axes[0, 3])
    # plot_bout(np.random.choice(sw_r), axes[1, 3])
    # plot_bout(np.random.choice(sw_r), axes[2, 3])
    plot_bout(np.random.choice(rfl), axes[0, 2])
    plot_bout(np.random.choice(rfl), axes[1, 2])
    plot_bout(np.random.choice(rfl), axes[2, 2])
    sns.despine(fig)
    fig.tight_layout()

    # plot example stabilizer movement
    stable_example_folder = "E:/Dropbox/2P_Data/H2BGc6s_MidHB_3Rep_161111"
    flist = os.listdir(stable_example_folder)
    flist = [f for f in flist if ".stable" in f]
    all_stable = [np.genfromtxt(stable_example_folder + "/" + f, delimiter='\t')[:, 3] for f in flist]
    sizes = [stb.size for stb in all_stable]
    all_stable = np.vstack([stb[None, :min(sizes)]-stb[0] for stb in all_stable])
    fig, ax = pl.subplots()
    sns.tsplot(all_stable, ax=ax, color="C1")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Z stabilizer offset [um]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Stabilizer_Movement.pdf", type="pdf")
