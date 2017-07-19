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


def plot_filter(mr: ModelResult, ax, ci=95, alpha=0.4):
    """
    Plots the model's filter with confidence intervals
    Args:
        mr: The ModelResult for which to plot the filter
        ax: Axis object on which to plot
        ci: The size of the confidence interval in percent to include
        alpha: The alpha value of the shaded error
    Returns:
        axis object
    """
    if ax is None:
        fig, ax = pl.subplots()
    p_lower = (100 - ci) / 2
    p_upper = 100 - p_lower
    e_lower = np.percentile(mr.trace_object["f"], p_lower, 0).ravel()[::-1]
    e_upper = np.percentile(mr.trace_object["f"], p_upper, 0).ravel()[::-1]
    m = np.mean(mr.trace_object["f"], 0).ravel()[::-1]
    time = np.arange(m.size) / 5
    ax.fill_between(time, e_lower, e_upper, alpha=alpha)
    ax.plot(time, m)
    return ax


def plot_lr_factors(factor_matrix, ax, ci=95):
    if ax is None:
        fig, ax = pl.subplots()
    p_lower = (100 - ci) / 2
    p_upper = 100 - p_lower
    e_lower = np.percentile(factor_matrix, p_lower, 0)
    e_upper = np.percentile(factor_matrix, p_upper, 0)
    m = np.mean(factor_matrix, 0)
    ix = np.arange(m.size)
    ax.bar(ix, m, color="C1")
    if m.size > 1:
        for i in range(m.size):
            ax.plot([ix[i], ix[i]], [e_lower[i], e_upper[i]], color="k")
    else:
        ax.plot([ix, ix], [e_lower, e_upper], color="k")
    max_val = max([np.max(e_upper), -1*np.min(e_lower)])
    ax.set_ylim(-max_val, max_val)
    return ax


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
    model_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/model_170702.hdf5', 'r')
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
        model_fit = model_results[n].predict_original()
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
        plot_filter(model_results[n], ax, 99)
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_filter.pdf", type="pdf")
        # coefficients
        fig, ax = pl.subplots()
        plot_lr_factors(model_results[n].trace_object["beta"][:, i], ax, 99)
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_lr_coefs.pdf", type="pdf")

    for i, n in enumerate(names_rh6):
        act_real = region_results["Rh6"].full_averages[:, i]
        act_real = standardize(trial_average(act_real)).ravel()
        model_fit = model_results[n].predict_original()
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
        plot_filter(model_results[n], ax, 99)
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_filter.pdf", type="pdf")
        # coefficients
        fig, ax = pl.subplots()
        if i == 1 or i == 3:
            plot_lr_factors(model_results[n].trace_object["beta"], ax, 99)
        else:
            beta = np.hstack((model_results[n].trace_object["beta_on"][:, None],
                              model_results[n].trace_object["beta_off"][:, None]))
            plot_lr_factors(beta, ax, 99)
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_lr_coefs.pdf", type="pdf")

    for i, n in enumerate(motor_res_names):
        act_real = motor_type_regs[:, i]
        model_fit = model_results[n].predict_original()
        # prediction
        fig, ax = pl.subplots()
        ax.scatter(model_fit, act_real, s=1)
        mn = min(model_fit.min(), act_real.min())
        ma = max(model_fit.max(), act_real.max())
        # add identity line
        ax.plot([mn, ma], [mn, ma], 'k--')
        ax.set_xlabel("Predicted rate [AU]")
        ax.set_ylabel("Real rate [AU]")
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_prediction.pdf", type="pdf")
        # coefficients
        fig, ax = pl.subplots()
        plot_lr_factors(model_results[n].trace_object["beta"], ax, 99)
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_lr_coefs.pdf", type="pdf")

    swim_fit = model_results["swim_out"].predict_original()
    # prediction
    fig, ax = pl.subplots()
    ax.scatter(swim_fit, swim_out, s=1)
    mn = min(swim_fit.min(), swim_out.min())
    ma = max(swim_fit.max(), swim_out.max())
    # add identity line
    ax.plot([mn, ma], [mn, ma], 'k--')
    ax.set_xlabel("Predicted rate [AU]")
    ax.set_ylabel("Real rate [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "swimout_prediction.pdf", type="pdf")
    # coefficients
    fig, ax = pl.subplots()
    plot_lr_factors(model_results["swim_out"].trace_object["beta"], ax, 99)
    sns.despine(fig, ax)
    fig.savefig(save_folder + "swimout_lr_coefs.pdf", type="pdf")

    flick_fit = model_results["flick_out"].predict_original()
    # prediction
    fig, ax = pl.subplots()
    ax.scatter(flick_fit, flick_out, s=1)
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
    plot_lr_factors(model_results["flick_out"].trace_object["beta"], ax, 99)
    sns.despine(fig, ax)
    fig.savefig(save_folder + "flickout_lr_coefs.pdf", type="pdf")

    # for illustration plot different stages of TG-ON fit
    to_plot = np.logical_and(trial_time > 55, trial_time < 120)
    fig, axes = pl.subplots(2, 2)
    axes[0, 0].plot(trial_time[to_plot], stim_in[to_plot])
    axes[0, 0].set_ylim(-1.5, 2)
    axes[0, 0].set_title("Stimulus")
    mr = model_results["TG_ON"]
    axes[0, 1].plot(trial_time[to_plot], mr.lr_result(mr.predictors)[to_plot])
    axes[0, 1].set_ylim(-1.5, 2)
    axes[0, 1].set_title("Linear result")
    axes[1, 0].plot(trial_time[to_plot], mr.filtered_result(mr.predictors)[to_plot])
    axes[1, 0].set_ylim(-1.5, 2)
    axes[1, 0].set_title("Convolved result")
    axes[1, 1].plot(trial_time[to_plot], mr.predict_original()[to_plot])
    axes[1, 1].set_ylim(-1.5, 2)
    axes[1, 1].set_title("Final result")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "model_steps_illus.pdf", type="pdf")
