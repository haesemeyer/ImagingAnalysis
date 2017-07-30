# Script to aggregate plots for Figure 4 of Heat-Imaging paper
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
import pandas


if __name__ == "__main__":
    save_folder = "./HeatImaging/Figure5/"
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
    # load motor output regressors
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "r")
    motor_type_regs = np.array(motor_store["motor_type_regs"])
    motor_store.close()
    # create motor containers
    tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
    nframes = region_results["Trigeminal"].full_averages.shape[0]
    itime = np.linspace(0, nframes / 5, nframes + 1)
    mc_swims = MotorContainer(sourceFiles, itime, exp_data[0].caTimeConstant, hdf5_store=tailstore,
                              predicate=unbiased_bouts)
    mc_flicks = MotorContainer(sourceFiles, itime, exp_data[0].caTimeConstant, tdd=mc_swims.tdd,
                               predicate=high_bias_bouts)
    motor_output = np.hstack((mc_swims.avg_motor_output[:, None], mc_flicks.avg_motor_output[:, None]))
    mo_ta = trial_average(motor_output.T).T

    # for trigeminal, Rh5/6 and Cerebellum, create regressor, reg-corr and 2-way linreg plots
    rep_time = np.arange(region_results["Trigeminal"].regressors.shape[0]) / 5.0
    detail_labels = ["Trigeminal", "Rh6", "Cerebellum"]
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

        # plot regressor-regressor correlations
        fig, ax = pl.subplots()
        sns.heatmap(np.corrcoef(ar.regressors.T), vmax=1, vmin=-1, center=0, annot=True, ax=ax)
        fig.savefig(save_folder + "regTOregCorrs_" + dl + ".pdf", type="pdf")

        # plot prediction of swims and flicks
        resmat_high = np.zeros((ar.regressors.shape[1], ar.regressors.shape[1]))
        resmat_low = np.zeros_like(resmat_high)
        for i in range(ar.regressors.shape[1]):
            for j in range(ar.regressors.shape[1]):
                if j < i:
                    resmat_high[i, j] = np.nan
                    resmat_low[i, j] = np.nan
                    continue
                regs = np.hstack((ar.regressors[:, i, None], ar.regressors[:, j, None]))
                lreg = LinearRegression()
                lreg.fit(regs, mo_ta[:, 1])
                resmat_high[i, j] = lreg.score(regs, mo_ta[:, 1])
                lreg = LinearRegression()
                lreg.fit(regs, mo_ta[:, 0])
                resmat_low[i, j] = lreg.score(regs, mo_ta[:, 0])
        fig, (ax_h, ax_l) = pl.subplots(ncols=2)
        sns.heatmap(resmat_high, 0, 1, cmap="RdBu_r", annot=True, ax=ax_h)
        sns.heatmap(resmat_low, 0, 1, cmap="RdBu_r", annot=True, ax=ax_l)
        fig.tight_layout()
        fig.savefig(save_folder + "motorPrediction_" + dl + ".pdf", type="pdf")

    # for all regions compute full-trace mutual information with sensory input and motor output
    mo_entropy = jknife_entropy(motor_output, 10)
    si_entropy = jknife_entropy(t_at_samp, 10)
    joint_ent = jknife_entropy(np.hstack((motor_output, t_at_samp[:, None])), 10)
    sens_motor_mi = mo_entropy + si_entropy - joint_ent
    motor_sys_entropy = jknife_entropy(motor_type_regs)
    joint_ent = jknife_entropy(np.hstack((motor_output, motor_type_regs)), 10)
    motor_motorsys_mi = mo_entropy + motor_sys_entropy - joint_ent
    joint_ent = jknife_entropy(np.hstack((t_at_samp[:, None], motor_type_regs)), 10)
    sens_motorsys_mi = si_entropy + motor_sys_entropy - joint_ent
    region_motor_mi = {"stimulus": sens_motor_mi, "motor cells": motor_motorsys_mi, "motor out": mo_entropy}
    region_sens_mi = {"motor out": sens_motor_mi, "motor cells": sens_motorsys_mi}
    for tl in test_labels:
        ar = region_results[tl]
        act_entropy = jknife_entropy(ar.full_averages, 10)
        m_jnt_entropy = jknife_entropy(np.hstack((ar.full_averages, motor_output)), 10)
        s_jnt_entropy = jknife_entropy(np.hstack((ar.full_averages, t_at_samp[:, None])), 10)
        region_motor_mi[tl] = mo_entropy + act_entropy - m_jnt_entropy
        region_sens_mi[tl] = si_entropy + act_entropy - s_jnt_entropy

    # plot motor mutual information for regions
    motor_mi = {k: [region_motor_mi[k]] for k in region_motor_mi.keys()}
    df = pandas.DataFrame(motor_mi)
    plot_order = ["stimulus"] + test_labels + ["motor cells", "motor out"]
    fig, ax = pl.subplots()
    sns.barplot(data=df, order=plot_order)
    ax.set_ylabel("Mutual information gain [bits]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "motor_MI.pdf", type="pdf")
