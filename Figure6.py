# Script to aggregate plots for Figure 6 of Heat-Imaging paper
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
import matplotlib as mpl
from Figure1 import trial_average
from analyzeSensMotor import RegionResults
from typing import Dict
from sensMotorModel import ModelResult, standardize, cubic_nonlin
import pandas


def col_std(m):
    avg = np.mean(m, 0, keepdims=True)
    s = np.std(m, 0, keepdims=True)
    return (m - avg) / s


if __name__ == "__main__":
    save_folder = "./HeatImaging/Figure6/"
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load region results
    test_labels = ["Trigeminal", "Rh6"]
    region_results = {}  # type: Dict[str, RegionResults]
    storage = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for k in test_labels:
        region_results[k] = (pickle.loads(np.array(storage[k])))
    storage.close()
    # load model results
    model_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/model.hdf5', 'r')
    model_results = pickle.loads(np.array(model_file["model_results"]))  # type: Dict[str, ModelResult]
    stim_in = np.array(model_file["stim_in"])
    model_file.close()
    names_tg = ["TG_ON", "TG_OFF"]
    names_rh6 = ["Fast_ON", "Slow_ON", "Fast_OFF", "Slow_OFF", "Delayed_OFF"]
    motor_res_names = ["M_All", "M_Flick", "M_Swim", "M_StimOn", "M_NoStim"]
    # load motor results
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "r")
    motor_type_regs = standardize(trial_average(np.array(motor_store["motor_type_regs"]).T)).T
    flick_out = standardize(trial_average(np.array(motor_store["flick_out"])))
    swim_out = standardize(trial_average(np.array(motor_store["swim_out"])))
    motor_store.close()
    trial_time = np.arange(region_results["Trigeminal"].regressors.shape[0]) / 5.0
    # for each trigeminal and rh6 model plot: Fit vs. real, and for insets: Impulse response and LR coefficients
    # for motor system fits plot: Fit vs. real and as inset: barplot of Rh6 coefficients
    for i, n in enumerate(names_tg):
        act_real = region_results["Trigeminal"].full_averages[:, i]
        act_real = standardize(trial_average(act_real)).ravel()
        model_fit = model_results[n].predict(stim_in)
        # prediction
        fig, ax = pl.subplots()
        ax.plot(trial_time, act_real, '--')
        ax.plot(trial_time, model_fit)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Activity [AU]")
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_prediction.pdf", type="pdf")
        # filter
        fig, ax = pl.subplots()
        ft = np.arange(model_results[n].filter_coefs.size) / 5
        ax.plot(ft, model_results[n].filter_coefs, 'k')
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_filter.pdf", type="pdf")
        # coefficients
        coefs = pandas.DataFrame({i: [f] for i, f in enumerate(model_results[n].lr_factors)})
        fig, ax = pl.subplots()
        sns.barplot(data=coefs, color=(150/255, 150/255, 150/255))
        ax.set_ylim(-np.max(np.abs(coefs.values)), np.max(np.abs(coefs.values)))
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_lr_coefs.pdf", type="pdf")

    tg_out = col_std(trial_average(region_results["Trigeminal"].full_averages.T).T)
    for i, n in enumerate(names_rh6):
        act_real = region_results["Rh6"].full_averages[:, i]
        act_real = standardize(trial_average(act_real)).ravel()
        model_fit = model_results[n].predict(tg_out)
        # prediction
        fig, ax = pl.subplots()
        ax.plot(trial_time, act_real, '--')
        ax.plot(trial_time, model_fit)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Activity [AU]")
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_prediction.pdf", type="pdf")
        # filter
        fig, ax = pl.subplots()
        ft = np.arange(model_results[n].filter_coefs.size) / 5
        ax.plot(ft, model_results[n].filter_coefs, 'k')
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_filter.pdf", type="pdf")
        # coefficients
        coefs = pandas.DataFrame({i: [f] for i, f in enumerate(model_results[n].lr_factors)})
        fig, ax = pl.subplots()
        sns.barplot(data=coefs, color=(150 / 255, 150 / 255, 150 / 255))
        ax.set_ylim(-np.max(np.abs(coefs.values)), np.max(np.abs(coefs.values)))
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_lr_coefs.pdf", type="pdf")

    rh6_out = col_std(trial_average(region_results["Rh6"].full_averages.T).T)
    for i, n in enumerate(motor_res_names):
        act_real = motor_type_regs[:, i]
        model_fit = model_results[n].predict(rh6_out)
        # prediction
        fig, ax = pl.subplots()
        ax.scatter(model_fit, act_real, s=2)
        mn = min(model_fit.min(), act_real.min())
        ma = max(model_fit.max(), act_real.max())
        # add identity line
        ax.plot([mn, ma], [mn, ma], 'k--')
        ax.set_xlabel("Predicted rate [AU]")
        ax.set_ylabel("Real rate [AU]")
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_prediction.pdf", type="pdf")
        # coefficients
        coefs = pandas.DataFrame({i: [f] for i, f in enumerate(model_results[n].lr_factors)})
        fig, ax = pl.subplots()
        sns.barplot(data=coefs, color=(150 / 255, 150 / 255, 150 / 255))
        ax.set_ylim(-np.max(np.abs(coefs.values)), np.max(np.abs(coefs.values)))
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_lr_coefs.pdf", type="pdf")

    swim_fit = model_results["swim_out"].predict(motor_type_regs)
    # prediction
    fig, ax = pl.subplots()
    ax.scatter(swim_fit, swim_out, s=2)
    mn = min(swim_fit.min(), swim_out.min())
    ma = max(swim_fit.max(), swim_out.max())
    # add identity line
    ax.plot([mn, ma], [mn, ma], 'k--')
    ax.set_xlabel("Predicted rate [AU]")
    ax.set_ylabel("Real rate [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "swimout_prediction.pdf", type="pdf")
    # coefficients
    coefs = pandas.DataFrame({i: [f] for i, f in enumerate(model_results["swim_out"].lr_factors)})
    fig, ax = pl.subplots()
    sns.barplot(data=coefs, color=(150 / 255, 150 / 255, 150 / 255))
    ax.set_ylim(-np.max(np.abs(coefs.values)), np.max(np.abs(coefs.values)))
    sns.despine(fig, ax)
    fig.savefig(save_folder + "swimout_lr_coefs.pdf", type="pdf")

    flick_fit = model_results["flick_out"].predict(motor_type_regs)
    # prediction
    fig, ax = pl.subplots()
    ax.scatter(flick_fit, flick_out, s=2)
    mn = min(flick_fit.min(), flick_out.min())
    ma = max(flick_fit.max(), flick_out.max())
    # add identity line
    ax.plot([mn, ma], [mn, ma], 'k--')
    ax.set_xlabel("Predicted rate [AU]")
    ax.set_ylabel("Real rate [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "flickout_prediction.pdf", type="pdf")
    # coefficients
    coefs = pandas.DataFrame({i: [f] for i, f in enumerate(model_results["flick_out"].lr_factors)})
    fig, ax = pl.subplots()
    sns.barplot(data=coefs, color=(150 / 255, 150 / 255, 150 / 255))
    ax.set_ylim(-np.max(np.abs(coefs.values)), np.max(np.abs(coefs.values)))
    sns.despine(fig, ax)
    fig.savefig(save_folder + "flickout_lr_coefs.pdf", type="pdf")
