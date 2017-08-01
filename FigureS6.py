# Script to plot panels in supplemental Figure 6
# Plot filter fit with noise simulation (original data, filtered data, real filter, derived filter)
# Plot nonlinearities of TG and Rh5/6 types
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
import matplotlib as mpl
from Figure1 import trial_average
from analyzeSensMotor import RegionResults
from typing import Dict
from sensMotorModel import ModelResult, standardize
import pymc3 as pm
from mh_2P import IndexingMatrix
from Figure6 import plot_lr_factors


def plot_nonlin(mresult: ModelResult, act_real: np.ndarray, ax: pl.Axes):
    if mresult.lr_factors.size == 1:
        linres = (mresult.predictors * mresult.lr_factors).ravel()
    else:
        linres = np.dot(mresult.predictors, mresult.lr_factors.T).ravel()
    filtered = np.convolve(linres, mresult.filter_coefs)[:linres.size]
    ax.scatter(filtered, act_real, s=1)
    nl_in = np.linspace(filtered.min(), filtered.max())
    ax.plot(nl_in, mresult.nonlin_function(nl_in, *mresult.nonlin_params), 'k')

if __name__ == "__main__":
    save_folder = "./HeatImaging/FigureS6/"
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
    mf_nofilt = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/model_noRh6Filts.hdf5', 'r')
    mr_nofilt = pickle.loads(np.array(mf_nofilt["model_results"]))  # type: Dict[str, ModelResult]
    mf_nofilt.close()
    names_tg = ["TG_ON", "TG_OFF"]
    names_rh6 = ["Fast_ON", "Slow_ON", "Fast_OFF", "Slow_OFF", "Delayed_OFF"]

    trial_time = np.arange(region_results["Trigeminal"].regressors.shape[0]) / 5.0

    # plot nonlinearity scatters
    for i, n in enumerate(names_tg):
        act_real = region_results["Trigeminal"].full_averages[:, i]
        act_real = standardize(trial_average(act_real)).ravel()
        fig, ax = pl.subplots()
        plot_nonlin(model_results[n], act_real, ax)
        ax.set_xlabel("Linear prediction [AU]")
        ax.set_ylabel("Observed activity [AU]")
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_nonlin.pdf", type="pdf")
    for i, n in enumerate(names_rh6):
        act_real = region_results["Rh6"].full_averages[:, i]
        act_real = standardize(trial_average(act_real)).ravel()
        fig, ax = pl.subplots()
        plot_nonlin(model_results[n], act_real, ax)
        ax.set_xlabel("Linear prediction [AU]")
        ax.set_ylabel("Observed activity [AU]")
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_nonlin.pdf", type="pdf")

    # modeling of filter derivation on noisy data - use hypothetical derivative on stim_in
    model_in = stim_in + np.random.randn(stim_in.size) * 0.2
    filter_real = np.zeros(22)
    filter_real[-1] = 1
    filter_real[-2] = -1
    filter_real /= np.linalg.norm(filter_real)
    model_out = standardize(np.convolve(stim_in, filter_real[::-1])[:stim_in.size]) + np.random.randn(stim_in.size)*0.2
    # clip off first value (filter transient)
    model_in = model_in[1:]
    model_out = model_out[1:]

    frames = np.arange(22).astype(float)[::-1][None, :]
    indexing_frames = np.arange(model_in.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, 22 - 1, 0, model_in.shape[0])
    model_input = model_in[ix]
    model_output = model_out[indexing_frames[cf:]][:, None]
    model = pm.Model()
    with model:
        # linear factors
        beta = pm.HalfNormal("beta", sd=2)
        # filter coefficients
        tau = pm.HalfNormal("tau", sd=10, shape=2)
        scale = pm.HalfNormal("scale", sd=5)
        # our noise
        sigma = pm.HalfNormal("sigma", sd=1)
        # create filter
        f = scale * frames * pm.math.exp(-frames / tau[0]) + (1 - frames) * pm.math.exp(-frames / tau[1])
        f = f / pm.math.sqrt(pm.math.sum(pm.math.sqr(f)))  # normalize the norm of the filter to one
        pm.Deterministic("f", f)
        # expected value of outcome
        linear_sum = beta * model_input
        mu = pm.math.sum(linear_sum * f, 1).ravel()
        # likelihood (sampling distributions) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=model_output.ravel())
    with model:
        trace = pm.sample(1000)

    linres = model_in * np.mean(trace["beta"])
    model_prediction = standardize(np.convolve(linres, np.mean(trace["f"], 0).ravel()[::-1])[:linres.size])

    # plot noisy true in and outputs as well as model fit
    fig, ax = pl.subplots()
    ax.plot(trial_time[1:], model_in, label="Input")
    ax.plot(trial_time[1:], model_out, label="Output")
    ax.plot(trial_time[1:], model_prediction, 'k', label="Model prediction")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activity [AU]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "simulation_in_out.pdf", type="pdf")

    # plot real and derived filter impulse response
    filter_time = np.arange(filter_real.size) / 5
    fig, ax = pl.subplots()
    ax.plot(filter_time, filter_real[::-1], label="Real filter kernel")
    ax.plot(filter_time, np.mean(trace["f"], 0).ravel()[::-1], 'k', label="Model derived filter kernel")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Filter coefficients")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "simulation_freal_fderived.pdf", type="pdf")

    # plot prediction / fit comparisons in Rh6 for comparison model without temporal filters
    for i, n in enumerate(names_rh6):
        act_real = region_results["Rh6"].full_averages[:, i]
        act_real = standardize(trial_average(act_real)).ravel()
        model_fit = mr_nofilt[n].predict_original()
        # prediction
        fig, ax = pl.subplots()
        ax.plot(trial_time, act_real, '--')
        ax.plot(trial_time, model_fit)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Activity [AU]")
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_nofilt_prediction.pdf", type="pdf")
        # coefficients
        fig, ax = pl.subplots()
        plot_lr_factors(mr_nofilt[n].trace_object["beta"], ax, 99)
        sns.despine(fig, ax)
        fig.savefig(save_folder + n + "_nofilt_lr_coefs.pdf", type="pdf")
