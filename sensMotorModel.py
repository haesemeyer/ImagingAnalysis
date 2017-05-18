# script to create model of sensory-motor transformations
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import RegionContainer, assign_region_label, MotorContainer, SLHRepeatExperiment
from multiExpAnalysis import get_stack_types, dff, max_cluster
from typing import List, Dict
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib as mpl
import pandas
from analyzeSensMotor import RegionResults
import os


default_exp_filter_length = 100  # default filter length for exponential decay is 20 seconds - external process
default_sin_filter_length = 20   # default filter length for neuronal filtering is 4 seconds


def sin_filter(f_shift, f_freq, f_decay):
    """
    Creates a filter that is described by a decaying sinusoidal function
    Args:
        f_shift: The time-shift (in frames) of the filter sinusoid
        f_freq: The frequency (in 1/Frames) of the filter sinusoid
        f_decay: The exponential decay of the filter

    Returns:
        The linear filter
    """
    frames = np.arange(default_sin_filter_length)
    return np.sin(frames * 2 * np.pi * f_freq + f_shift) * np.exp(-f_decay * frames)


def sin_filter_fit_function(x, f_shift, f_freq, f_decay):
    """
    Function to fit a decaying sinusoidal input filter
    Args:
        x: The input data
        f_shift: The time-shift (in frames) of the filter sinusoid
        f_freq: The frequency (in 1/Frames) of the filter sinusoid
        f_decay: The exponential decay of the filter

    Returns:
        The filtered signal
    """
    f = sin_filter(f_shift, f_freq, f_decay)
    return np.convolve(x, f)[:x.size]


def exp_filter(f_scale, f_decay):
    """
    Creates a filter that is described by an exponential decay
    Args:
        f_scale: The scale of the exponential
        f_decay: The decay constant of the exponential 

    Returns:
        The linear filter
    """
    frames = np.arange(default_exp_filter_length)
    return f_scale * np.exp(-f_decay * frames)


def exp_filter_fit_function(x, f_scale, f_decay):
    """
    Function to fit an exponential input filter
    Args:
        x: The input data
        f_scale: The scale of the exponential
        f_decay: The decay constant of the exponential 

    Returns:
        The filtered signal
    """

    f = exp_filter(f_scale, f_decay)
    return np.convolve(x, f)[:x.size]


def cubic_nonlin(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d


def exp_nonlin(x, offset, rate, scale):
    return scale*np.exp(rate * x) + offset


def predict_response_w_filter(predictors, output, param_bounds=([-10, -0.1, -0.5], [10, 2.5, 10])):
    # linear regression fit
    lr = LinearRegression()
    lr.fit(predictors, output)
    # fit linear filter
    reg_out = lr.predict(predictors)
    shift, freq, decay = curve_fit(sin_filter_fit_function, reg_out, output, bounds=param_bounds)[0]
    return lr, (shift, freq, decay)


def r2(prediction, real):
    ss_res = np.sum((prediction - real)**2)
    ss_tot = np.sum((real - np.mean(real))**2)
    return 1 - ss_res/ss_tot


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


if __name__ == "__main__":
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    no_nan_aa = np.array(dfile['no_nan_aa'])
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
    # load region sensory motor results
    result_labels = ["Trigeminal", "Rh6", "Cerebellum", "Habenula", "Pallium", "SubPallium", "POA"]
    region_results = {}  # type: Dict[str, RegionResults]
    analysis_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for rl in result_labels:
        region_results[rl] = pickle.loads(np.array(analysis_file[rl]))
    analysis_file.close()
    # create motor containers if necessary
    if os.path.exists('H:/ClusterLocations_170327_clustByMaxCorr/motor_output.hdf5'):
        motor_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/motor_output.hdf5', 'r')
        flicks_out = np.array(motor_file["flicks_out"])
        swims_out = np.array(motor_file["swims_out"])
        motor_file.close()
    else:
        n_frames = region_results["Trigeminal"].full_averages.shape[0]
        itime = np.linspace(0, n_frames / 5, n_frames + 1)
        tailstore = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/taildata.hdf5', 'r')
        cat = exp_data[0].caTimeConstant
        mc_flicks = MotorContainer(sourceFiles, itime, cat, predicate=high_bias_bouts, hdf5_store=tailstore)
        mc_swims = MotorContainer(sourceFiles, itime, cat, predicate=unbiased_bouts, tdd=mc_flicks.tdd)
        flicks_out = mc_flicks.avg_motor_output
        swims_out = mc_swims.avg_motor_output
        motor_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/motor_output.hdf5', 'w')
        motor_file.create_dataset("flicks_out", data=flicks_out)
        motor_file.create_dataset("swims_out", data=swims_out)
        motor_file.close()

    # NOTE: We are unable to explain difference in some OFF responses in very beginning vs. inter-stimulus periods.
    # Therefore all r2 will be calculated *ignoring* the period before the first stim-on (first 300 frames) and fits
    # of nonlinearities will also ignore these frames.
    # Also to make coefficients more comparable all model inputs and desired outputs
    # will be scaled to unit variance and mean-subtracted
    f_s = 300

    # provisionally use the convolved laser stimulus as model input
    stim_in = exp_data[0].stimOn
    stim_in = standardize(stim_in)

    # Laser input to trigeminal ON type
    tg_on = region_results["Trigeminal"].full_averages[:, 0]
    tg_on = standardize(tg_on)
    # 1) Linear regression
    lr = LinearRegression()
    lr.fit(stim_in[:, None], tg_on)
    print("ON coefficients = ", lr.coef_)
    reg_out = lr.predict(stim_in[:, None])
    # 2) Fit linear filter - since most of this will be governed by Laser -> Temperature use exponential filter
    scale, decay = curve_fit(exp_filter_fit_function, reg_out, tg_on)[0]
    filtered_out = exp_filter_fit_function(reg_out, scale, decay)
    # 3) Fit cubic output nonlinearity
    a, b, c, d = curve_fit(cubic_nonlin, filtered_out[f_s:], tg_on[f_s:])[0]
    tg_on_prediction = cubic_nonlin(filtered_out, a, b, c, d)
    # plot successive fits
    fig, ax = pl.subplots()
    ax.plot(stim_in, lw=0.5)
    ax.plot(reg_out, lw=0.5)
    ax.plot(filtered_out, lw=0.75)
    ax.plot(tg_on_prediction, lw=1.5)
    ax.plot(tg_on, lw=1.5)
    ax.set_title("Successive predictions Trigeminal ON type from stimulus")
    sns.despine(fig, ax)
    # plot linear input filter
    fig, ax = pl.subplots()
    ax.plot(np.arange(-99, 1) / 5, exp_filter(scale, decay)[::-1], 'k')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("f(t)")
    ax.set_ylim(0)
    ax.set_title("Linear filter, Trigeminal ON")
    sns.despine(fig, ax)
    # plot output nonlinearity
    fig, ax = pl.subplots()
    input_range = np.linspace(filtered_out.min(), filtered_out.max())
    ax.scatter(filtered_out[f_s:], tg_on[f_s:], alpha=0.2, s=1, color='C0')
    ax.plot(input_range, cubic_nonlin(input_range, a, b, c, d), 'k')
    ax.set_xlabel("f(Temperature)")
    ax.set_ylabel("g[f(Temperature)]")
    ax.set_title("Output nonlinearity, Trigeminal ON")
    sns.despine(fig, ax)
    print("R2 TG ON prediction = ", r2(tg_on_prediction[f_s:], tg_on[f_s:]))

    # Laser input to trigeminal OFF type
    tg_off = region_results["Trigeminal"].full_averages[:, 1]
    # 1) Linear regression
    lr = LinearRegression()
    lr.fit(stim_in[:, None], tg_off)
    print("OFF coefficients = ", lr.coef_)
    reg_out = lr.predict(stim_in[:, None])
    # 2) Fit linear filter - since most of this will be governed by Laser -> Temperature use exponential filter
    scale, decay = curve_fit(exp_filter_fit_function, reg_out, tg_off)[0]
    filtered_out = exp_filter_fit_function(reg_out, scale, decay)
    # 3) Fit exponential output nonlinearity
    o, r, s = curve_fit(exp_nonlin, filtered_out[f_s:], tg_off[f_s:])[0]
    tg_off_prediction = exp_nonlin(filtered_out, o, r, s)
    # plot successive fits
    fig, ax = pl.subplots()
    ax.plot(stim_in, lw=0.5)
    ax.plot(reg_out, lw=0.5)
    ax.plot(filtered_out, lw=0.75)
    ax.plot(tg_off_prediction, lw=1.5)
    ax.plot(tg_off, lw=1.5)
    ax.set_title("Successive predictions Trigeminal OFF type from stimulus")
    sns.despine(fig, ax)
    # plot linear input filter
    fig, ax = pl.subplots()
    ax.plot(np.arange(-99, 1) / 5, exp_filter(scale, decay)[::-1], 'k')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("f(t)")
    ax.set_ylim(0)
    ax.set_title("Linear filter, Trigeminal OFF")
    sns.despine(fig, ax)
    # plot output nonlinearity
    fig, ax = pl.subplots()
    input_range = np.linspace(filtered_out.min(), filtered_out.max())
    ax.scatter(filtered_out[f_s:], tg_off[f_s:], alpha=0.2, s=1, color='C0')
    ax.plot(input_range, exp_nonlin(input_range, o, r, s), 'k')
    ax.set_xlabel("f(Temperature)")
    ax.set_ylabel("g[f(Temperature)]")
    ax.set_title("Output nonlinearity, Trigeminal OFF")
    sns.despine(fig, ax)
    print("R2 TG OFF prediction = ", r2(tg_off_prediction[f_s:], tg_off[f_s:]))
