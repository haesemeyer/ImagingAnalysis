import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import RegionContainer, assign_region_label, MotorContainer, SLHRepeatExperiment, IndexingMatrix, CaConvolve
from multiExpAnalysis import get_stack_types, dff, max_cluster
from typing import List, Dict
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, least_squares
import matplotlib as mpl
from analyzeSensMotor import RegionResults, trial_average
import os
import pymc3 as pm


def standardize(x):
    """
    Removes the mean from the input and scales variance to unity
    """
    return (x - np.mean(x)) / np.std(x)


def on_off_filter(tau_on, tau_off, filter_len=100):
    """
    Creates a double-exponential ris-decay filter
    Args:
        tau_on: The time-constant of the on-component
        tau_off: The time-constant of the off-component
        filter_len: The length of the filter in frames

    Returns:
        The linear filter
    """
    frames = np.arange(filter_len)
    return np.exp(-frames / tau_off) * (1 - np.exp(-frames / tau_on))


def make_stim_tg_model(stim_in, tg_activity, filter_len=100):
    indexing_frames = np.arange(20, stim_in.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, filter_len - 1, 0, stim_in.shape[0])
    model_in = stim_in[ix]
    model_out = tg_activity[indexing_frames[cf:]][:, None]
    frames = np.arange(filter_len).astype(float)[::-1][None, :]
    conv_model = pm.Model()
    with conv_model:
        # linear factor
        beta = pm.Normal("beta", mu=0, sd=2)
        # filter coefficients
        tau_on = pm.HalfNormal("tau_on", sd=200)
        tau_off = pm.HalfNormal("tau_off", sd=200)
        f = pm.math.exp(-frames / tau_off) * (1 - pm.math.exp(-frames / tau_on))
        pm.Deterministic("f", f)
        # our noise
        sigma = pm.HalfNormal("sigma", sd=1)
        # expected value of outcome
        linear_sum = beta * model_in
        mu = pm.math.sum(linear_sum * f, 1).ravel()
        # likelihood (sampling distributions) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=model_out.ravel())
    return conv_model


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

    model_results = {}

    # load temperature
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    t_at_samp = np.array(stim_file["sine_L_H_temp"])
    t_at_samp = trial_average(np.add.reduceat(t_at_samp, np.arange(0, t_at_samp.size, 20 // 5))).ravel() / (20 // 5)
    stim_file.close()
    m_in, s_in = np.mean(t_at_samp), np.std(t_at_samp)  # store for later use
    stim_in = standardize(t_at_samp)

    # Laser input to trigeminal ON type
    tg_on = trial_average(region_results["Trigeminal"].full_averages[:, 0])
    tg_on = standardize(tg_on)
