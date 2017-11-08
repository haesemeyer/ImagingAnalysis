# script to create model of sensory-motor transformations
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


default_exp_filter_length = 100  # default filter length for exponential decay is 20 seconds - external process
default_sin_filter_length = 10  # default filter length for neuronal filtering is 2 seconds


class ModelResult:
    """
    Stores result of a modeling step
    """
    def __init__(self, lr_inputs: np.ndarray, lr_factors, filter_coefs: np.ndarray, nonlin_type: str, nonlin_params):
        """
        Creates a new model result
        Args:
            lr_inputs: The original input activities to the model
            lr_factors: The linear regression factors acting on the inputs
            filter_coefs: The filter coefficients for convolution or None if no filtering
            nonlin_type: The type of the output nonlinearity or None if not used
            nonlin_params: The parameters of the output nonlinearity
        """
        if lr_inputs.shape[1] != lr_factors.size:
            raise ValueError("The needs to be one lr_factor per lr_input!")
        if nonlin_type is None:
            self.nonlin_type = None
            self.nonlin_params = None
            self.nonlin_function = None
        elif nonlin_type.upper() == "CUBIC":
            self.nonlin_type = "CUBIC"
            self.nonlin_params = nonlin_params
            self.nonlin_function = cubic_nonlin
        elif nonlin_type.upper() == "EXP":
            self.nonlin_type = "EXP"
            self.nonlin_params = nonlin_params
            self.nonlin_function = exp_nonlin
        elif nonlin_type.upper() == "SIG":
            self.nonlin_type = "SIG"
            self.nonlin_params = nonlin_params
            self.nonlin_function = sig_nonlin
        else:
            raise ValueError("Did not recognize nonlin type. Should be 'CUBIC' or 'EXP' or 'SIG'")
        self.predictors = lr_inputs
        self.lr_factors = lr_factors
        self.filter_coefs = filter_coefs
        self.trace_object = None

    def lr_result(self, model_in):
        if self.lr_factors.size > 1:
            return np.dot(model_in, self.lr_factors.T).ravel()
        else:
            return (model_in * self.lr_factors).ravel()

    def filtered_result(self, model_in):
        lr_res = self.lr_result(model_in)
        if self.filter_coefs is None:
            return lr_res
        else:
            return np.convolve(lr_res, self.filter_coefs)[:lr_res.size]

    def predict_original(self):
        """
        Predicts the output of this modeling step using the original inputs
        Returns:
            The predicted timeseries
        """
        return self.predict(self.predictors)

    def predict(self, model_in):
        """
        Predicts the output of this modeling step given the input
        Args:
            model_in: n_samples x n_features input to the model

        Returns:
            The predicted timeseries
        """
        if self.nonlin_type is None:
            return self.filtered_result(model_in)
        else:
            return self.nonlin_function(self.filtered_result(model_in), *self.nonlin_params)


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


def on_off_filter(tau_on, tau_off):
    """
    Creates a double-exponential ris-decay filter
    Args:
        tau_on: The time-constant of the on-component
        tau_off: The time-constant of the off-component

    Returns:
        The linear filter
    """
    frames = np.arange(default_exp_filter_length)
    return np.exp(-frames / tau_off) * (1 - np.exp(-frames / tau_on))


def on_off_filter_fit_function(x, tau_on, tau_off):
    """
    Function to fit an on-off filter
    Args:
        x: The input data
        tau_on: The time-constant of the on-component
        tau_off: The time-constant of the off-component

    Returns:
        The filtered signal
    """
    f = on_off_filter(tau_on, tau_off)
    return np.convolve(x, f)[:x.size]


def cubic_nonlin(x, a, b, c, d):
    """
    Parametrization of a cubic nonlinearity applied to x
    """
    return a*(x**3) + b*(x**2) + c*x + d


def sig_nonlin(x, s, tau, dt, o):
    """
    Parametrization of a sigmoid nonlinearity applied to x
    Args:
        x: The input
        s: The scale of the sigmoid
        tau: The timescale of the transition
        dt: The position of the halfway point
        o: Offset term

    Returns:
        The sigmoid transformation of x
    """
    return s * (1 / (1+np.exp(-tau*(x-dt))) + o)


def exp_nonlin(x, offset, rate, scale):
    """
    Parametrization of an exponential nonlinearity applied to x
    """
    return scale*np.exp(rate * x) + offset


def r2(prediction, real):
    """
    Computes coefficient of determination
    """
    ss_res = np.sum((prediction - real)**2)
    ss_tot = np.sum((real - np.mean(real))**2)
    return 1 - ss_res/ss_tot


def fvu(prediction, real):
    """
    Computes the fraction of unexplained variance
    """
    return np.var(prediction-real) / np.var(real)


def standardize(x):
    """
    Removes the mean from the input and scales variance to unity
    """
    return (x - np.mean(x)) / np.std(x)


def dexp_f(frames: np.ndarray, s1, t1, t2) -> np.ndarray:
    """
    Returns a filter that is the sum of an exponential and it's derivate
    Args:
        frames: Frames array (usually 0...n) over which to calculate filter
        s1: The scaling factor of the exponential
        t1: The time-constant of the exponential
        t2: The time-constant of the derivative

    Returns:
        The filter coefficients
    """
    # NOTE: The actual derivative would multiply the second term by 1/t2 - however this results
    # in unstable behavior returning filters that are not fittable
    return s1 * frames * np.exp(-frames/t1) + 1 * (np.exp(-frames/t2) - frames*np.exp(-frames/t2))


def make_dexp_residual_function(inputs: np.ndarray, output: np.ndarray, f_len):
    """
    Create function to calculate residuals over a time-filtered auto-regressive model
    Args:
        inputs: n_timepoints x n_cells array of inputs to consider
        output: The desired timeseries output
        f_len: The length of the double-exponential filter in frames

    Returns:
        A function to be used for least_squares that calculates model-output residuals
    """
    def residuals(x):
        nonlocal model_in
        nonlocal model_out
        nonlocal f_len
        s1, t1, t2 = x[:3]  # the filter parameters
        alphas = x[3:]
        # create filter - note: In our indexing matrices for the stimuli the most recent element [t0] will be in the
        # *last* column of the matrix. For fitting our filter therefore needs to be inverted.
        frames = np.arange(f_len)
        f = dexp_f(frames, s1, t1, t2)
        f = f[::-1][None, :]
        prediction = np.zeros(model_out.size)
        for j, m in enumerate(model_in):
            prediction += alphas[j] * np.sum(m * f, 1)
        return model_out - prediction
    model_in = []
    if inputs.ndim != 2:
        raise ValueError("inputs has to be 2D array even if only one cell is present")
    if inputs.shape[0] != output.size:
        raise ValueError("The number of timepoints in inputs and output needs to be the same")
    indexing_frames = np.arange(20, inputs.shape[0])
    ix, cf, cb = IndexingMatrix(indexing_frames, f_len-1, 0, inputs.shape[0])
    for i in range(inputs.shape[1]):
        model_in.append(inputs[:, i][ix])
    model_out = output[indexing_frames[cf:]]
    p0 = np.ones(3 + inputs.shape[1])
    return residuals, p0


def run_model(laser_stimulus, model_results: Dict[str, ModelResult], exclude=None, noTGFilter=False):
    """
    Run model on stimulus predicting from input to motor output
    Args:
        laser_stimulus: The stimulus temperature, standardized
        model_results: Dictionary of model fits
        exclude: Outputs of types listed in excluded will be set to 0
        noTGFilter: If set to true, inputs to TG units won't be filtered by the TG kernel

    Returns:
        [0]: Swim prediction (conv. with Ca kernel by model)
        [1]: Flick prediction (conv. with Ca kernel by model)
        [2]: Prediction of activity in Rh6
    """
    if exclude is None:
        exclude = []
    if noTGFilter:
        fsum = np.sum(model_results["TG_ON"].filter_coefs)
        tg_on_prediction = model_results["TG_ON"].lr_result(laser_stimulus)
        tg_on_prediction *= fsum
    else:
        tg_on_prediction = model_results["TG_ON"].predict(laser_stimulus)
    if "TG_ON" in exclude:
        tg_on_prediction[:] = 0
    if noTGFilter:
        fsum = np.sum(model_results["TG_OFF"].filter_coefs)
        tg_off_prediction = model_results["TG_OFF"].lr_result(laser_stimulus)
        tg_off_prediction *= fsum
    else:
        tg_off_prediction = model_results["TG_OFF"].predict(laser_stimulus)
    if "TG_OFF" in exclude:
        tg_off_prediction[:] = 0
    tg_out_prediction = np.hstack((tg_on_prediction[:, None], tg_off_prediction[:, None]))
    # first the slow Rh6 types which are created via direct input from the trigeminal types
    slow_on_prediction = model_results["Slow_ON"].predict(tg_out_prediction)
    if "Slow_ON" in exclude:
        slow_on_prediction[:] = 0
    slow_off_prediction = model_results["Slow_OFF"].predict(tg_out_prediction)
    if "Slow_OFF" in exclude:
        slow_off_prediction[:] = 0
    off_inh_out = np.hstack((tg_on_prediction[:, None], slow_off_prediction[:, None]))
    fast_on_prediction = model_results["Fast_ON"].predict(off_inh_out)
    if "Fast_ON" in exclude:
        fast_on_prediction[:] = 0
    fast_off_prediction = model_results["Fast_OFF"].predict(off_inh_out)
    if "Fast_OFF" in exclude:
        fast_off_prediction[:] = 0
    on_inh_out = np.hstack((slow_on_prediction[:, None], tg_off_prediction[:, None]))
    del_off_prediction = model_results["Delayed_OFF"].predict(on_inh_out)
    if "Delayed_OFF" in exclude:
        del_off_prediction[:] = 0
    rh6_out_prediction = np.hstack((fast_on_prediction[:, None], slow_on_prediction[:, None],
                                    fast_off_prediction[:, None], slow_off_prediction[:, None],
                                    del_off_prediction[:, None]))
    m_all_p = model_results["M_All"].predict(rh6_out_prediction)
    if "M_All" in exclude:
        m_all_p[:] = 0
    m_fl_p = model_results["M_Flick"].predict(rh6_out_prediction)
    if "M_Flick" in exclude:
        m_fl_p[:] = 0
    m_sw_p = model_results["M_Swim"].predict(rh6_out_prediction)
    if "M_Swim" in exclude:
        m_sw_p[:] = 0
    m_so_p = model_results["M_StimOn"].predict(rh6_out_prediction)
    if "M_StimOn" in exclude:
        m_so_p[:] = 0
    m_ns_p = model_results["M_NoStim"].predict(rh6_out_prediction)
    if "M_NoStim" in exclude:
        m_ns_p[:] = 0
    motor_out_prediction = np.hstack((m_all_p[:, None], m_fl_p[:, None], m_sw_p[:, None], m_so_p[:, None],
                                      m_ns_p[:, None]))
    swim_prediction = model_results["swim_out"].predict(motor_out_prediction)
    flick_prediction = model_results["flick_out"].predict(motor_out_prediction)
    return swim_prediction, flick_prediction, rh6_out_prediction


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
    # 1) Linear regression
    lr = LinearRegression()
    lr.fit(stim_in[:, None], tg_on)
    print("ON coefficients = ", lr.coef_)
    reg_out = lr.predict(stim_in[:, None])
    # 2) Fit linear filter - since most of this will be governed by Laser -> Temperature use exponential filter
    t_on, t_off = curve_fit(on_off_filter_fit_function, reg_out, tg_on)[0]
    filtered_out = on_off_filter_fit_function(reg_out, t_on, t_off)
    # 3) Fit cubic output nonlinearity
    a, b, c, d = curve_fit(cubic_nonlin, filtered_out, tg_on)[0]
    tg_on_prediction = cubic_nonlin(filtered_out, a, b, c, d)
    # plot fit
    fig, ax = pl.subplots()
    ax.plot(stim_in, lw=0.5)
    ax.plot(filtered_out, lw=0.75)
    ax.plot(tg_on_prediction, lw=1.5)
    ax.plot(tg_on, lw=1.5)
    ax.set_title("Successive predictions Trigeminal ON type from stimulus")
    sns.despine(fig, ax)
    # plot linear input filter
    fig, ax = pl.subplots()
    ax.plot(np.arange(-99, 1) / 5, on_off_filter(t_on, t_off)[::-1], 'k')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("f(t)")
    ax.set_ylim(0)
    ax.set_title("Linear filter, Trigeminal ON")
    sns.despine(fig, ax)
    # plot output nonlinearity
    fig, ax = pl.subplots()
    input_range = np.linspace(filtered_out.min(), filtered_out.max())
    ax.scatter(filtered_out, tg_on, alpha=0.2, s=1, color='C0')
    ax.plot(input_range, cubic_nonlin(input_range, a, b, c, d), 'k')
    ax.set_xlabel("f(Temperature)")
    ax.set_ylabel("g[f(Temperature)]")
    ax.set_title("Output nonlinearity, Trigeminal ON")
    sns.despine(fig, ax)
    print("R2 TG ON prediction = ", r2(tg_on_prediction, tg_on))
    model_results["TG_ON"] = ModelResult(stim_in[:, None], lr.coef_, on_off_filter(t_on, t_off), "CUBIC", (a, b, c, d))

    # Laser input to trigeminal OFF type
    tg_off = trial_average(region_results["Trigeminal"].full_averages[:, 1])
    tg_off = standardize(tg_off)
    # 1) Linear regression
    lr = LinearRegression()
    lr.fit(stim_in[:, None], tg_off)
    print("OFF coefficients = ", lr.coef_)
    reg_out = lr.predict(stim_in[:, None])
    # 2) Fit linear filter - since most of this will be governed by Laser -> Temperature use exponential filter
    t_on, t_off = curve_fit(on_off_filter_fit_function, reg_out, tg_off)[0]
    filtered_out = on_off_filter_fit_function(reg_out, t_on, t_off)
    # 3) Fit exponential output nonlinearity
    a, b, c, d = curve_fit(cubic_nonlin, filtered_out, tg_off)[0]
    tg_off_prediction = cubic_nonlin(filtered_out, a, b, c, d)
    # plot successive fits
    fig, ax = pl.subplots()
    ax.plot(stim_in, lw=0.5)
    ax.plot(filtered_out, lw=0.75)
    ax.plot(tg_off_prediction, lw=1.5)
    ax.plot(tg_off, lw=1.5)
    ax.set_title("Successive predictions Trigeminal OFF type from stimulus")
    sns.despine(fig, ax)
    # plot linear input filter
    fig, ax = pl.subplots()
    ax.plot(np.arange(-99, 1) / 5, on_off_filter(t_on, t_off)[::-1], 'k')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("f(t)")
    ax.set_ylim(0)
    ax.set_title("Linear filter, Trigeminal OFF")
    sns.despine(fig, ax)
    # plot output nonlinearity
    fig, ax = pl.subplots()
    input_range = np.linspace(filtered_out.min(), filtered_out.max())
    ax.scatter(filtered_out, tg_off, alpha=0.2, s=1, color='C0')
    ax.plot(input_range, cubic_nonlin(input_range, a, b, c, d), 'k')
    ax.set_xlabel("f(Temperature)")
    ax.set_ylabel("g[f(Temperature)]")
    ax.set_title("Output nonlinearity, Trigeminal OFF")
    sns.despine(fig, ax)
    print("R2 TG OFF prediction = ", r2(tg_off_prediction, tg_off))
    model_results["TG_OFF"] = ModelResult(stim_in[:, None], lr.coef_, on_off_filter(t_on, t_off), "CUBIC", (a, b, c, d))

    # fit of slow Rh6 units from trigeminal inputs
    filter_length = 22
    tg_out = np.hstack((tg_on[:, None], tg_off[:, None]))
    response_names = ["Fast_ON", "Slow_ON", "Fast_OFF", "Slow_OFF", "Delayed_OFF"]
    fig, ax = pl.subplots()
    filter_time = np.arange(filter_length) / -5.0

    for i in [1, 3]:
        output = standardize(trial_average(region_results["Rh6"].full_averages[:, i]))
        resid_fun, p0 = make_dexp_residual_function(tg_out, output, filter_length)
        # for trigeminal since cells are glutamatergic only allow positive activations
        tg_bounds_upper = np.full(p0.size, np.Inf)
        tg_bounds_lower = np.full(p0.size, -np.Inf)
        tg_bounds_lower[-2] = 0
        tg_bounds_lower[-1] = 0
        params = least_squares(resid_fun, p0/10, bounds=(tg_bounds_lower, tg_bounds_upper)).x
        print("Type {0} coefficients: {1}".format(i, params[-tg_out.shape[1]:]))
        f = dexp_f(np.arange(filter_length), *params[:-2])
        ax.plot(filter_time, f)
        lr_sum = np.sum(tg_out * params[-tg_out.shape[1]:][None, :], 1)
        filtered_out = np.convolve(lr_sum, f)[:tg_on_prediction.size]
        nl_params = curve_fit(cubic_nonlin, filtered_out, output)[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        n = response_names[i]
        model_results[n] = ModelResult(tg_out, params[-tg_out.shape[1]:], f, "CUBIC", nl_params)

    # fit of fastON and fastOFF Rh6 types as they both require OFF type inhibition
    rh6_slow_off = standardize(trial_average(region_results["Rh6"].full_averages[:, 3]))[:, None]
    off_inh_out = np.hstack((tg_on[:, None], rh6_slow_off))
    for i in [0, 2]:
        output = standardize(trial_average(region_results["Rh6"].full_averages[:, i]))
        resid_fun, p0 = make_dexp_residual_function(off_inh_out, output, filter_length)
        params = least_squares(resid_fun, p0/10).x
        print("Type {0} coefficients: {1}".format(i, params[-tg_out.shape[1]:]))
        f = dexp_f(np.arange(filter_length), *params[:-2])
        ax.plot(filter_time, f)
        lr_sum = np.sum(off_inh_out * params[-off_inh_out.shape[1]:][None, :], 1)
        filtered_out = np.convolve(lr_sum, f)[:tg_on_prediction.size]
        nl_params = curve_fit(cubic_nonlin, filtered_out, output)[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        n = response_names[i]
        model_results[n] = ModelResult(off_inh_out, params[-off_inh_out.shape[1]:], f, "CUBIC", nl_params)

    # fit of delayed OFF type which requires ON type inhibition
    rh6_slow_on = standardize(trial_average(region_results["Rh6"].full_averages[:, 1]))[:, None]
    on_inh_out = np.hstack((rh6_slow_on, tg_off[:, None]))
    for i in [4]:
        output = standardize(trial_average(region_results["Rh6"].full_averages[:, i]))
        resid_fun, p0 = make_dexp_residual_function(on_inh_out, output, filter_length)
        params = least_squares(resid_fun, p0/10).x
        print("Type {0} coefficients: {1}".format(i, params[-tg_out.shape[1]:]))
        f = dexp_f(np.arange(filter_length), *params[:-2])
        ax.plot(filter_time, f)
        lr_sum = np.sum(on_inh_out * params[-on_inh_out.shape[1]:][None, :], 1)
        filtered_out = np.convolve(lr_sum, f)[:tg_on_prediction.size]
        nl_params = curve_fit(cubic_nonlin, filtered_out, output)[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        n = response_names[i]
        model_results[n] = ModelResult(on_inh_out, params[-on_inh_out.shape[1]:], f, "CUBIC", nl_params)

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
        lr = LinearRegression()
        lr.fit(rh_6_out, output)
        prediction = lr.predict(rh_6_out)
        model_results[n] = ModelResult(rh_6_out, lr.coef_, None, None, None)
        print("Type {0} coefficients: {1}".format(i, lr.coef_))
        print("Type {0} R2: {1}".format(i, r2(prediction, output)))
        # test contribution of Rh6 components alone
        for j in range(rh_6_out.shape[1]):
            lr = LinearRegression()
            lr.fit(rh_6_out[:, j][:, None], output)
            red_pred = lr.predict(rh_6_out[:, j][:, None])
            print("Type {0} with rh6 component {1} R2: {2}".format(i, j, r2(red_pred, output)))

    # predict swims
    lr = LinearRegression()
    lr.fit(motor_type_regs, swim_out)
    prediction = lr.predict(motor_type_regs)
    model_results["swim_out"] = ModelResult(motor_type_regs, lr.coef_, None, None, None)
    print("Swim coefficients: {0}".format(lr.coef_))
    print("Swim R2: {0}".format(r2(prediction, swim_out)))
    for j in range(motor_type_regs.shape[1]):
        lr = LinearRegression()
        lr.fit(motor_type_regs[:, j][:, None], swim_out)
        red_pred = lr.predict(motor_type_regs[:, j][:, None])
        print("Swim with motor type component {0} R2: {1}".format(j, r2(red_pred, swim_out)))

    # predict flicks
    lr = LinearRegression()
    lr.fit(motor_type_regs, flick_out)
    prediction = lr.predict(motor_type_regs)
    model_results["flick_out"] = ModelResult(motor_type_regs, lr.coef_, None, None, None)
    print("Flick coefficients: {0}".format(lr.coef_))
    print("Flick R2: {0}".format(r2(prediction, flick_out)))
    for j in range(motor_type_regs.shape[1]):
        lr = LinearRegression()
        lr.fit(motor_type_regs[:, j][:, None], flick_out)
        red_pred = lr.predict(motor_type_regs[:, j][:, None])
        print("Flick with motor type component {0} R2: {1}".format(j, r2(red_pred, flick_out)))

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
                xticklabels=["Fast ON", "Slow ON", "Fast OFF", "Slow OFF", "Dld. OFF"], ax=ax)
    # plot cluster boundaries
    covered = 0
    for i in range(pred_reg_corr_mat.shape[1]):
        covered += np.sum(mclust.labels_ == i)
        ax.plot([0, dt_sig_corrs.shape[1] + 1], [mclust.labels_.size - covered, mclust.labels_.size - covered], 'k')
    ax.set_ylabel("Cells")
