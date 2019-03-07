# script to create model of sensory-motor transformations using pymc3
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import MotorContainer, SLHRepeatExperiment, IndexingMatrix
from multiExpAnalysis import max_cluster
from typing import List, Dict
from motorPredicates import unbiased_bouts, high_bias_bouts
from scipy.optimize import curve_fit
import matplotlib as mpl
from analyzeSensMotor import RegionResults, trial_average
import os
import pymc3 as pm
from sensMotorModel import cubic_nonlin, r2, ModelResult, standardize, run_model


def make_firstTier_Rh6_model(tg_on, tg_off, rh6_activity, filter_len=22):
    """
    Creates a model to relate trigeminal output activity to rh6_activity with only positive
    linear factors.
    Args:
        tg_on: ON activity in the trigeminal
        tg_off: OFF activity in the trigeminal
        rh6_activity: The rh6 activity to predict
        filter_len: The length of the double-exponential filter

    Returns:
        A pymc3 model that can be used for sampling/prediction
    """
    indexing_frames = np.arange(tg_on.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, filter_len - 1, 0, tg_on.shape[0])
    on_in = tg_on[ix]
    off_in = tg_off[ix]
    model_out = rh6_activity[indexing_frames[cf:]][:, None]
    # frames = np.arange(filter_len).astype(float)[::-1][None, :]
    frames = np.arange(filter_len).astype(float)[::-1][:, None]
    model = pm.Model()
    with model:
        # linear factors
        beta = pm.HalfNormal("beta", sd=2, shape=2)
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
        linear_sum = beta[0] * on_in + beta[1] * off_in
        # mu = pm.math.sum(linear_sum * f, 1).ravel()
        mu = pm.math.dot(linear_sum, f).ravel()
        # likelihood (sampling distributions) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=model_out.ravel())
    return model


def make_offInhibited_Rh6_model(tg_on, slow_off, rh6_activity, filter_len=22):
    """
        Creates a model to relate trigeminal and rh6 slow off activity to rh6_activity with inhibition from the
        slow_off type
        Args:
            tg_on: ON activity in the trigeminal
            slow_off: Activity of the rh6 slow off type
            rh6_activity: The rh6 activity to predict
            filter_len: The length of the double-exponential filter

        Returns:
            A pymc3 model that can be used for sampling/prediction
    """
    indexing_frames = np.arange(tg_on.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, filter_len - 1, 0, tg_on.shape[0])
    on_in = tg_on[ix]
    off_in = slow_off[ix]
    model_out = rh6_activity[indexing_frames[cf:]][:, None]
    frames = np.arange(filter_len).astype(float)[::-1][:, None]
    model = pm.Model()
    with model:
        # linear factors
        beta_on = pm.HalfNormal("beta_on", sd=2)
        beta_off = pm.Normal("beta_off", mu=0, sd=2)  # allow inhibition from rh6 type!
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
        linear_sum = beta_on * on_in + beta_off * off_in
        mu = pm.math.dot(linear_sum, f).ravel()
        # likelihood (sampling distributions) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=model_out.ravel())
    return model


def make_onInhibited_Rh6_model(slow_on, tg_off, rh6_activity, filter_len=22):
    """
        Creates a model to relate trigeminal off and rh6 slow on activity to rh6_activity with inhibition from the
        slow_on type
        Args:
            slow_on: Activity of the rh6 slow on type
            tg_off: Activity of the trigeminal off type
            rh6_activity: The rh6 activity to predict
            filter_len: The length of the double-exponential filter

        Returns:
            A pymc3 model that can be used for sampling/prediction
    """
    indexing_frames = np.arange(tg_on.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, filter_len - 1, 0, tg_on.shape[0])
    on_in = slow_on[ix]
    off_in = tg_off[ix]
    model_out = rh6_activity[indexing_frames[cf:]][:, None]
    frames = np.arange(filter_len).astype(float)[::-1][:, None]
    model = pm.Model()
    with model:
        # linear factors
        beta_off = pm.HalfNormal("beta_off", sd=2)
        beta_on = pm.Normal("beta_on", mu=0, sd=2)  # allow inhibition from rh6 type!
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
        linear_sum = beta_on * on_in + beta_off * off_in
        mu = pm.math.dot(linear_sum, f).ravel()
        # likelihood (sampling distributions) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=model_out.ravel())
    return model


def make_linearModel(act_in, act_out, constrained=False):
    """
    Creates a linear regression model for fitting
    Args:
        act_in: The input activity
        act_out: The desired output
        constrained: If true only allow positive coefficients
    Returns:
        A pymc3 model that can be used for sampling/prediction
    """
    model = pm.Model()
    with model:
        if constrained:
            beta = pm.HalfNormal("beta", sd=2, shape=act_in.shape[1])
        else:
            beta = pm.Normal("beta", mu=0, sd=2, shape=act_in.shape[1])
        sigma = pm.HalfNormal("sigma", sd=1)
        # expected value of outcome
        mu = pm.math.dot(act_in, beta.T)
        # likelihood
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=act_out.ravel())
    return model


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

    model_results = {}  # type: Dict[str, ModelResult]

    n_samples = 1000

    # load temperature
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    t_at_samp = np.array(stim_file["sine_L_H_temp"])
    t_at_samp = trial_average(np.add.reduceat(t_at_samp, np.arange(0, t_at_samp.size, 20 // 5))).ravel() / (20 // 5)
    stim_file.close()
    m_in, s_in = np.mean(t_at_samp), np.std(t_at_samp)  # store for later use
    stim_in = standardize(t_at_samp)

    # fit both trigeminal types concurrently, sharing the filter
    tg_on = trial_average(region_results["Trigeminal"].full_averages[:, 0])
    tg_on = standardize(tg_on)
    tg_off = trial_average(region_results["Trigeminal"].full_averages[:, 1])
    tg_off = standardize(tg_off)
    frames = np.arange(100).astype(float)[::-1][None, :]
    indexing_frames = np.arange(tg_on.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, 100 - 1, 0, tg_on.shape[0])
    stimulus_in = stim_in[ix]
    model_out = np.hstack((tg_on[indexing_frames[cf:]][:, None], tg_off[indexing_frames[cf:]][:, None]))
    tg_model = pm.Model()
    with tg_model:
        beta = pm.Normal("beta", mu=0, sd=2, shape=2)
        tau_on = pm.HalfNormal("tau_on", sd=5)
        tau_off = pm.HalfNormal("tau_off", sd=200)
        f = pm.math.exp(-frames / tau_off) * (1 - pm.math.exp(-frames / tau_on))
        f = f / pm.math.sqrt(pm.math.sum(pm.math.sqr(f)))  # normalize the norm of the filter to one
        pm.Deterministic("f", f)
        sigma = pm.HalfNormal("sigma", sd=0.1, shape=2)
        mu1 = pm.math.sum((beta[0] * stimulus_in) * f, 1).ravel()
        mu2 = pm.math.sum((beta[1] * stimulus_in) * f, 1).ravel()
        mu = pm.math.stack([mu1, mu2], 1)
        Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=model_out)
    start = pm.find_MAP(model=tg_model)
    with tg_model:
        tg_trace = pm.sample(n_samples, start=start)

    beta = np.mean(tg_trace["beta"], 0).ravel()
    reg_out = stim_in * beta[0]
    f = np.mean(tg_trace["f"], 0).ravel()
    filtered_out = np.convolve(reg_out, f[::-1])[:reg_out.size]
    # Fit cubic output nonlinearity
    a, b, c, d = curve_fit(cubic_nonlin, filtered_out, tg_on)[0]
    tg_on_prediction = cubic_nonlin(filtered_out, a, b, c, d)
    print("TG ON factor = ", beta[0])
    print("R2 TG ON prediction = ", r2(tg_on_prediction, tg_on))
    model_results["TG_ON"] = ModelResult(stim_in[:, None], beta[0], f[::-1], "CUBIC", (a, b, c, d))
    model_results["TG_ON"].trace_object = tg_trace
    reg_out = stim_in * beta[1]
    filtered_out = np.convolve(reg_out, f[::-1])[:reg_out.size]
    # Fit cubic output nonlinearity
    a, b, c, d = curve_fit(cubic_nonlin, filtered_out, tg_off)[0]
    tg_off_prediction = cubic_nonlin(filtered_out, a, b, c, d)
    print("TG OFF factor = ", beta[1])
    print("R2 TG OFF prediction = ", r2(tg_off_prediction, tg_off))
    model_results["TG_OFF"] = ModelResult(stim_in[:, None], beta[1], f[::-1], "CUBIC", (a, b, c, d))
    model_results["TG_OFF"].trace_object = tg_trace

    # Trigeminal output to Rh6 first tier
    filter_length = 22
    tg_out = np.hstack((tg_on[:, None], tg_off[:, None]))
    response_names = ["Fast_ON", "Slow_ON", "Fast_OFF", "Slow_OFF", "Delayed_OFF"]
    filter_time = np.arange(filter_length) / -5.0
    for i in [1, 3]:
        output = standardize(trial_average(region_results["Rh6"].full_averages[:, i]))
        model = make_linearModel(tg_out, output, True)
        start = pm.find_MAP(model=model)
        with model:
            trace = pm.sample(n_samples, start=start)
        betas = np.mean(trace["beta"], 0)
        reg_out = np.dot(tg_out, betas.T)
        filtered_out = reg_out
        nl_params = curve_fit(cubic_nonlin, filtered_out, output)[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        print("Type {0} coefficients: {1}".format(i, betas))
        n = response_names[i]
        model_results[n] = ModelResult(tg_out, betas, None, "CUBIC", nl_params)
        model_results[n].trace_object = trace

    # fit of fastON and fastOFF Rh6 types as they both require OFF type inhibition
    rh6_slow_off = standardize(trial_average(region_results["Rh6"].full_averages[:, 3]))
    off_inh_out = np.hstack((tg_on[:, None], rh6_slow_off[:, None]))
    for i in [0, 2]:
        output = standardize(trial_average(region_results["Rh6"].full_averages[:, i]))
        model = make_linearModel(off_inh_out, output)
        start = pm.find_MAP(model=model)
        with model:
            trace = pm.sample(n_samples, start=start)
        betas = np.mean(trace["beta"], 0)
        reg_out = np.dot(off_inh_out, betas.T)
        filtered_out = reg_out
        nl_params = curve_fit(cubic_nonlin, filtered_out, output)[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        print("Type {0} coefficients: {1}".format(i, betas))
        n = response_names[i]
        model_results[n] = ModelResult(off_inh_out, betas, None, "CUBIC", nl_params)
        model_results[n].trace_object = trace

    # fit of delayed OFF type which requires ON type inhibition
    rh6_slow_on = standardize(trial_average(region_results["Rh6"].full_averages[:, 1]))
    on_inh_out = np.hstack((rh6_slow_on[:, None], tg_off[:, None]))
    for i in [4]:
        output = standardize(trial_average(region_results["Rh6"].full_averages[:, i]))
        model = make_linearModel(on_inh_out, output)
        start = pm.find_MAP(model=model)
        with model:
            trace = pm.sample(n_samples, start=start)
        betas = np.mean(trace["beta"], 0)
        reg_out = np.dot(on_inh_out, betas.T)
        filtered_out = reg_out
        nl_params = curve_fit(cubic_nonlin, filtered_out, output)[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        print("Type {0} coefficients: {1}".format(i, betas))
        n = response_names[i]
        model_results[n] = ModelResult(on_inh_out, betas, None, "CUBIC", nl_params)
        model_results[n].trace_object = trace

    # fit of motor type rates from Rh6 cells - since we do not fit activity traces but rates do not fit filters
    motor_store = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/motor_system.hdf5", "r")
    motor_type_regs = standardize(trial_average(np.array(motor_store["motor_type_regs"]).T)).T
    flick_out = standardize(trial_average(np.array(motor_store["flick_out"])))
    swim_out = standardize(trial_average(np.array(motor_store["swim_out"])))
    motor_store.close()
    rh_6_out = standardize(trial_average(region_results["Rh6"].full_averages.T)).T
    motor_res_names = ["M_All", "M_Flick", "M_Swim", "M_StimOn", "M_NoStim"]
    for i in range(motor_type_regs.shape[1]):
        n = motor_res_names[i]
        output = motor_type_regs[:, i]
        model = make_linearModel(rh_6_out, output)
        with model:
            trace = pm.sample(n_samples)
        betas = np.mean(trace["beta"], 0)
        prediction = np.dot(rh_6_out, betas.T)
        model_results[n] = ModelResult(rh_6_out, betas, None, None, None)
        model_results[n].trace_object = trace
        print("Type {0} coefficients: {1}".format(i, betas))
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))

    # predict swims
    model = make_linearModel(motor_type_regs, swim_out)
    with model:
        trace_swim = pm.sample(n_samples)
    betas = np.mean(trace_swim["beta"], 0)
    prediction = np.dot(motor_type_regs, betas.T)
    model_results["swim_out"] = ModelResult(motor_type_regs, betas, None, None, None)
    model_results["swim_out"].trace_object = trace_swim
    print("Swim coefficients: {0}".format(betas))
    print("Swim R2: {0}".format(r2(prediction, swim_out)))

    # predict flicks
    model = make_linearModel(motor_type_regs, flick_out)
    with model:
        trace_flick = pm.sample(n_samples)
    betas = np.mean(trace_flick["beta"], 0)
    prediction = np.dot(motor_type_regs, betas.T)
    model_results["flick_out"] = ModelResult(motor_type_regs, betas, None, None, None)
    model_results["flick_out"].trace_object = trace_flick
    print("Flick coefficients: {0}".format(betas))
    print("Flick R2: {0}".format(r2(prediction, flick_out)))

    swim_pred, flick_pred = run_model(stim_in, model_results)[:2]
    trial_time = np.arange(stim_in.size) / 5.0
    fig, (ax_sw, ax_flk) = pl.subplots(ncols=2, sharex=True, sharey=True)
    ax_sw.plot(trial_time, standardize(swim_out), 'k', label="Swims")
    ax_sw.plot(trial_time, standardize(swim_pred), "C0", label="Swim prediction")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Motor output [AU]")
    ax_sw.set_title("R2 = {0:.2}".format(np.corrcoef(swim_pred, swim_out)[0, 1]**2))
    ax_sw.legend()
    ax_flk.plot(trial_time, standardize(flick_out), 'k', label="Flicks")
    ax_flk.plot(trial_time, standardize(flick_pred), "C1", label="Flick prediction")
    ax_flk.set_xlabel("Time [s]")
    ax_flk.set_title("R2 = {0:.2}".format(np.corrcoef(flick_pred, flick_out)[0, 1] ** 2))
    ax_flk.legend()
    sns.despine(fig)
    fig.tight_layout()

    # try to predict motor output during detail-char experiments
    stim_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/stimFile.hdf5', 'r')
    dt_t_at_samp = np.array(stim_file["detail_char_temp"])
    dt_t_at_samp = trial_average(np.add.reduceat(dt_t_at_samp,
                                                 np.arange(0, dt_t_at_samp.size, 20 // 5)), 10).ravel() / (20 // 5)
    stim_file.close()
    # use the same subtraction/division as used for the temperature stimulus above *not* this mean and std!
    lc = (dt_t_at_samp - m_in) / s_in
    s, f, dt_rh6 = run_model(lc, model_results)
    # use last trial prediction - since the average is the same
    dt_swim_pred = s[-675:]
    dt_flick_pred = f[-675:]
    detChar_swims = np.load("detailChar_swims.npy")
    detChar_flicks = np.load("detailChar_flicks.npy")
    dt_trial_time = np.arange(dt_swim_pred.size) / 5.0
    # we can only predict what happens during periods where there is no tap influence
    no_tap_inf = np.logical_and(dt_trial_time > 10, dt_trial_time < 128)
    fig, (ax_sw, ax_flk) = pl.subplots(ncols=2, sharex=True, sharey=True)
    ax_sw.plot(dt_trial_time[no_tap_inf], standardize(detChar_swims[no_tap_inf]), 'k', label="Swims")
    ax_sw.plot(dt_trial_time[no_tap_inf], standardize(dt_swim_pred[no_tap_inf]), "C0", label="Swim prediction")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Motor output [AU]")
    ax_sw.set_title("R2 = {0:.2}".format(np.corrcoef(detChar_swims[no_tap_inf], dt_swim_pred[no_tap_inf])[0, 1] ** 2))
    ax_sw.legend()
    ax_flk.plot(dt_trial_time[no_tap_inf], standardize(detChar_flicks[no_tap_inf]), 'k', label="Flicks")
    ax_flk.plot(dt_trial_time[no_tap_inf], standardize(dt_flick_pred[no_tap_inf]), "C1", label="Flick prediction")
    ax_flk.set_xlabel("Time [s]")
    ax_flk.set_title("R2 = {0:.2}".format(np.corrcoef(detChar_flicks[no_tap_inf], dt_flick_pred[no_tap_inf])[0, 1] ** 2))
    ax_flk.legend()
    sns.despine(fig)
    fig.tight_layout()

    # use predicted rh6 activity in detail char experiments to cluster Rh6 data from those experiments into our types
    dfile = h5py.File("H:/ClusterLocations_170327_clustByMaxCorr/detailChar_data.hdf5", "r")
    dt_act = np.array(dfile["all_activity"])
    dt_regions = pickle.loads(np.array(dfile["all_rl_pickle"]))
    dfile.close()
    rh6_act = dt_act[(dt_regions == "Rh_6").ravel(), :]
    # create trial averages of rh6 activity
    ta_rh6_act = np.mean(rh6_act.reshape((rh6_act.shape[0], 25, rh6_act.shape[1] // 25)), 1)
    # create correlation matrix for correlations of real activity to predicted rh6 activity as regressors
    pred_reg_corr_mat = np.zeros((ta_rh6_act.shape[0], dt_rh6.shape[1]))
    for i in range(dt_rh6.shape[1]):
        reg = dt_rh6[-675:, i]
        for j, a in enumerate(ta_rh6_act):
            pred_reg_corr_mat[j, i] = np.corrcoef(a, reg)[0, 1]
    dt_sig_corrs = pred_reg_corr_mat[np.sum(pred_reg_corr_mat >= 0.6, 1) > 0, :]
    activity_sig_corrs = ta_rh6_act[np.sum(pred_reg_corr_mat >= 0.6, 1) > 0, :]
    mclust = max_cluster(np.argmax(dt_sig_corrs, 1))
    fig, ax = pl.subplots()
    sns.heatmap(dt_sig_corrs[np.argsort(mclust.labels_), :], yticklabels=50,
                xticklabels=["Fast ON", "Slow ON", "Fast OFF", "Slow OFF", "Dld. OFF"],
                ax=ax, vmax=1, vmin=-1, center=0)
    # plot cluster boundaries
    covered = 0
    for i in range(pred_reg_corr_mat.shape[1]):
        covered += np.sum(mclust.labels_ == i)
        ax.plot([0, dt_sig_corrs.shape[1] + 1], [mclust.labels_.size - covered, mclust.labels_.size - covered], 'k')
    ax.set_ylabel("Cells")
