# Script to aggregate plots for  Supplemental Figure 5 of Heat-Imaging paper
# Plot Number of identified neurons regular versus shuffle for regions
# Plot mutual information with *sensory stimulus* for each cluster average before/after shuffle

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
import matplotlib as mpl
from Figure1 import trial_average
from mh_2P import MotorContainer, SLHRepeatExperiment, CaConvolve
from analyzeSensMotor import RegionResults, jknife_entropy
from typing import Dict, List
from sklearn.linear_model import LinearRegression
from motorPredicates import high_bias_bouts, unbiased_bouts
from pandas import DataFrame, Series


if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS5/"
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
    dfile.close()
    # load stimulus
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    t_at_samp = np.array(stim_file["sine_L_H_temp"])
    t_at_samp = np.add.reduceat(t_at_samp, np.arange(0, t_at_samp.size, 20 // 5)) / (20 // 5)
    t_at_samp = CaConvolve(t_at_samp, exp_data[0].caTimeConstant, 5)
    # remove convolution transient in beginning
    t_at_samp[:74] = t_at_samp[74]
    stim_file.close()
    # load region sensori-motor results
    test_labels = ["Trigeminal", "Rh6", "Rh2", "Cerebellum", "Habenula", "Pallium", "SubPallium", "POA"]
    region_results = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for k in test_labels:
        region_results[k] = (pickle.loads(np.array(storage[k])))
    storage.close()
    # load shuffled sensori-motor results
    rr_shuffled = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata_sh.hdf5', 'r')
    for k in test_labels:
        rr_shuffled[k] = (pickle.loads(np.array(storage[k])))
    storage.close()

    # Plot cluster averages of forebrain regions
    rep_time = np.arange(region_results["Trigeminal"].regressors.shape[0]) / 5.0
    detail_labels = ["Habenula", "Pallium", "SubPallium", "POA"]
    for dl in detail_labels:
        ar = region_results[dl]
        # plot trial-averaged cluster activity
        fig, ax = pl.subplots()
        for i, c in enumerate(ar.regs_clust_labels):
            cname = "C{0}".format(int(c))
            data = trial_average(ar.region_acts[ar.region_mem == c, :])
            sns.tsplot(data=data, time=rep_time, color=cname, ax=ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("dF/F")
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder+"regressors_"+dl+".pdf", type="pdf")

    # Compare fraction of cluster-membership for different regions between real and shuffled data
    counts_real = {k: np.sum(region_results[k].region_mem > -1) for k in test_labels}
    counts_sh = {k: np.sum(rr_shuffled[k].region_mem > -1) for k in test_labels}
    dframe = DataFrame({k: [counts_sh[k] / counts_real[k] * 100] for k in test_labels})

    fig, ax = pl.subplots()
    sns.barplot(data=dframe, ax=ax, order=test_labels)
    ax.set_ylabel("Fraction of cells after shuffling [%]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "pos_fraction_compare" + ".pdf", type="pdf")

    # Compute the mutual information between stimulus and each original/shuffled cluster for comparison
    si_entropy = jknife_entropy(t_at_samp, 10)
    sens_real_mi = []
    sens_sh_mi = []
    for tl in test_labels:
        ar = region_results[tl]
        for i in range(ar.full_averages.shape[1]):
            ent = jknife_entropy(ar.full_averages[:, i], 10)
            jent = jknife_entropy(np.hstack((t_at_samp[:, None], ar.full_averages[:, i][:, None])), 10)
            sens_real_mi.append(si_entropy + ent - jent)
        ar = rr_shuffled[tl]
        if ar.full_averages.size < 1:
            continue
        for i in range(ar.full_averages.shape[1]):
            ent = jknife_entropy(ar.full_averages[:, i], 10)
            jent = jknife_entropy(np.hstack((t_at_samp[:, None], ar.full_averages[:, i][:, None])), 10)
            sens_sh_mi.append(si_entropy + ent - jent)

    d = {"Real": sens_real_mi, "Shuffled": sens_sh_mi}
    dframe_mi = DataFrame(dict([(k, Series(d[k])) for k in d]))
    fig, ax = pl.subplots()
    sns.barplot(data=dframe_mi, ax=ax, order=["Real", "Shuffled"])
    ax.set_ylabel("Sensory mutual information [bits]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "real_sh_sensmi_compare" + ".pdf", type="pdf")
