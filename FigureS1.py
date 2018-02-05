# Script to aggregate plots for  Supplemental Figure 1 of Heat-Imaging paper


import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib as mpl
import h5py
import numpy as np
from mh_2P import MotorContainer, SLHRepeatExperiment
import pickle
from typing import List, Dict
import os
from multiExpAnalysis import dff
from Figure1 import trial_average
from analyzeSensMotor import RegionResults

if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS1/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    no_nan_aa = np.array(dfile['no_nan_aa'])
    pstream = np.array(dfile['exp_data_pickle'])
    exp_data = pickle.loads(pstream)  # type: List[SLHRepeatExperiment]
    del pstream
    # limit sourceFiles to the contents of all_activity
    sourceFiles = [(g[0], e.original_time_per_frame) for e in exp_data for g in e.graph_info]
    sourceFiles = [sf for i, sf in enumerate(sourceFiles) if no_nan_aa[i]]
    all_activity = np.array(dfile["all_activity"])
    nframes = all_activity.shape[1]
    membership = np.array(dfile["membership"])
    mship_nonan = membership[no_nan_aa]
    dfile.close()
    # load temperature stimulus
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    t_at_samp = np.array(stim_file["sine_L_H_temp"])
    t_at_samp = np.add.reduceat(t_at_samp, np.arange(0, t_at_samp.size, 20 // 5)) / (20 // 5)
    stim_file.close()
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

    # load shuffled data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/df_shuffle_170615.hdf5', 'r')
    sh_membership = np.array(dfile['membership'])
    sh_no_nan_aa = np.array(dfile['no_nan_aa'])
    sh_mship_nonan = sh_membership[sh_no_nan_aa]
    sh_all_activity = np.array(dfile["all_activity"])
    dfile.close()

    sh_active_cells = np.logical_and(sh_mship_nonan > -1, sh_mship_nonan < 6)

    # plot heatmap of shuffled activity sorted by membership
    mship_active = sh_mship_nonan[sh_active_cells]
    act_to_plot = dff(trial_average(sh_all_activity[sh_active_cells, :]))
    fig, ax = pl.subplots()
    sns.heatmap(act_to_plot[np.argsort(mship_active), :], xticklabels=150, yticklabels=250, vmin=-3, vmax=3, ax=ax,
                rasterized=True)
    fig.savefig(save_folder + "sh_activity_heatmap.pdf", type="pdf")

    # plot heatmap of RFP stabilized data sorted by membership
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/RFP_stabilized.hdf5', 'r')
    rfp_mem = np.array(dfile['membership'])
    rfp_nonanaa = np.array(dfile['no_nan_aa'])
    rfp_memnn = rfp_mem[rfp_nonanaa]
    rfp_act = np.array(dfile["all_activity"])
    dfile.close()
    rfp_active = np.logical_and(rfp_memnn > -1, rfp_memnn < 6)
    mship_active = rfp_memnn[rfp_active]
    act_to_plot = dff(trial_average(rfp_act[rfp_active, :], n_trials=2))  # these expts. only have 2 trials
    act_to_plot_gcamp = dff(trial_average(all_activity, n_trials=3))
    fig, ax = pl.subplots()
    sns.heatmap(act_to_plot[np.argsort(mship_active), :], xticklabels=250, yticklabels=250, vmin=-3, vmax=3, ax=ax,
                rasterized=True)
    fig.savefig(save_folder + "rfp_stable_activity_heatmap.pdf", type="pdf")

    # plot Gcamp and RFP cluster averages
    time = np.arange(act_to_plot.shape[1]) / 5.0
    fig, axes = pl.subplots(nrows=6, ncols=2, sharex=True, sharey=True)
    for i, c in enumerate(np.unique(mship_active)):
        sns.tsplot(act_to_plot[mship_active == c, :], time, ax=axes[i, 1], color="C3")
        if i < 4:
            sns.tsplot(act_to_plot_gcamp[mship_nonan == c, :], time, ax=axes[i, 0], color="C2")
        else:
            sns.tsplot(act_to_plot_gcamp[mship_nonan == c, :], time, ax=axes[i, 0], color="C4")
        axes[i, 0].set_ylabel("dF/F0")
    axes[5, 0].set_xlabel("Time [s]")
    axes[5, 1].set_xlabel("Time [s]")
    axes[5, 0].set_xticks([0, 30, 60, 90, 120, 150])
    axes[5, 1].set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "rfp_stable_activity_clusterAvgs.pdf", type="pdf")

    # plot region example calcium traces
    test_labels = ["Trigeminal", "Rh6", "Pallium"]
    region_results = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for k in test_labels:
        region_results[k] = (pickle.loads(np.array(storage[k])))
    storage.close()
    # plot trigeminal example activity
    rr = region_results["Trigeminal"]
    fig, axes = pl.subplots(nrows=2, ncols=2, sharex=True)
    axes = axes.ravel()
    axes[0].plot(time, trial_average(rr.region_acts[rr.region_mem == 0, :])[10, :], "C2")
    axes[1].plot(time, trial_average(rr.region_acts[rr.region_mem == 0, :])[50, :], "C2")
    axes[2].plot(time, trial_average(rr.region_acts[rr.region_mem == 3, :])[6, :], "C4")
    axes[3].plot(time, trial_average(rr.region_acts[rr.region_mem == 3, :])[58, :], "C4")
    for i, a in enumerate(axes):
        if i > 1:
            a.set_xlabel("Trial time [s]")
        if i % 2 == 0:
            a.set_ylabel("dF/F0")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "TG_Example_Acts.pdf", type="pdf")
    # plot rh5/6 example activity
    rr = region_results["Rh6"]
    fig, axes = pl.subplots(nrows=2, ncols=2, sharex=True)
    axes = axes.ravel()
    axes[0].plot(time, trial_average(rr.region_acts[rr.region_mem == 0, :])[13, :], "C2")
    axes[1].plot(time, trial_average(rr.region_acts[rr.region_mem == 1, :])[50, :], "C2")
    axes[2].plot(time, trial_average(rr.region_acts[rr.region_mem == 3, :])[6, :], "C4")
    axes[3].plot(time, trial_average(rr.region_acts[rr.region_mem == 5, :])[30, :], "C4")
    for i, a in enumerate(axes):
        if i > 1:
            a.set_xlabel("Trial time [s]")
        if i % 2 == 0:
            a.set_ylabel("dF/F0")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "Rh56_Example_Acts.pdf", type="pdf")
    # plot pallium example activity
    rr = region_results["Pallium"]
    fig, axes = pl.subplots(nrows=2, ncols=2, sharex=True)
    axes = axes.ravel()
    axes[0].plot(time, trial_average(rr.region_acts[rr.region_mem == 0, :])[13, :], "C2")
    axes[1].plot(time, trial_average(rr.region_acts[rr.region_mem == 3, :])[5, :], "C2")
    axes[2].plot(time, trial_average(rr.region_acts[rr.region_mem == 4, :])[6, :], "C4")
    axes[3].plot(time, trial_average(rr.region_acts[rr.region_mem == 5, :])[30, :], "C4")
    for i, a in enumerate(axes):
        if i > 1:
            a.set_xlabel("Trial time [s]")
        if i % 2 == 0:
            a.set_ylabel("dF/F0")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "Pallium_Example_Acts.pdf", type="pdf")

    # plot detail char behavioral output split into swims and flicks as well as ON OFF activity heatmap
    dfile = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/detailChar_data.hdf5", "r")
    avg_swim = trial_average(np.array(dfile['swim_raw']), n_trials=25).ravel() * 5
    avg_flick = trial_average(np.array(dfile['flick_raw']), n_trials=25).ravel() * 5
    rep_time = np.array(dfile["rep_time"])
    dt_act = np.array(dfile["all_activity"])
    stim_units = np.array(dfile["stim_units"])
    act_sign = np.array(dfile["act_sign"])
    dfile.close()

    fig, (ax_heat, ax_tap) = pl.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})
    ax_heat.plot(rep_time[rep_time < 125], avg_swim[rep_time < 125])
    ax_heat.plot(rep_time[rep_time < 125], avg_flick[rep_time < 125])
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("Bout frequency [Hz]")
    ax_heat.set_xticks([0, 30, 60, 90, 120])
    ax_tap.plot(rep_time[rep_time >= 125], avg_swim[rep_time >= 125])
    ax_tap.plot(rep_time[rep_time >= 125], avg_flick[rep_time >= 125])
    ax_tap.set_xlabel("Time [s]")
    ax_tap.set_ylabel("Bout frequency [Hz]")
    ax_tap.set_xticks([125, 130, 135])
    # ax_tap.plot([129.8, 129.8], [0, 5], "k--")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "dtchar_motor_output.pdf", type="pdf")


    def dff(ts):
        # f0 = np.percentile(ts, 10, axis=1, keepdims=True)
        f0 = np.mean(ts[:, 10 * 5:20 * 5], axis=1, keepdims=True)
        f0[f0 < 0.05] = 0.05
        return (ts - f0) / f0
    act_to_plot = dff(trial_average(dt_act[stim_units, :], n_trials=25))
    fig, ax = pl.subplots()
    sns.heatmap(act_to_plot[np.argsort(act_sign[stim_units])[::-1], :], xticklabels=150, yticklabels=100, vmin=-2.5,
                vmax=2.5, ax=ax, rasterized=True)
    fig.savefig(save_folder + "dtchar_activity_heatmap.pdf", type="pdf")

    # NOTE: Trial-to-trial correlation plots are created by "analyzeTrialCorrelations.py" script
