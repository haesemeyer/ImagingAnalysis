# script to create model of sensory-motor transformations
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
from mh_2P import RegionContainer, assign_region_label, MotorContainer, SLHRepeatExperiment, IndexingMatrix
from multiExpAnalysis import get_stack_types, dff, max_cluster
from typing import List, Dict
from motorPredicates import left_bias_bouts, right_bias_bouts, unbiased_bouts, high_bias_bouts
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, least_squares
import matplotlib as mpl
from analyzeSensMotor import RegionResults
import os


default_exp_filter_length = 100  # default filter length for exponential decay is 20 seconds - external process
default_sin_filter_length = 10  # default filter length for neuronal filtering is 2 seconds


class ModelResult:
    """
    Stores result of a modeling step
    """
    def __init__(self, lr_inputs: np.ndarray, filter_type: str, fit_params, nonlin_type: str, nonlin_params):
        """
        Creates a new model result
        Args:
            lr_inputs: The input activities to the model
            filter_type: The type of the linear filter or None if not used
            fit_params: The parameters (coefficients and filter) of the performed fit
            nonlin_type: The type of the output nonlinearity or None if not used
            nonlin_params: The parameters of the output nonlinearity
        """
        if filter_type is None:
            self.filter_type = None
        elif filter_type.upper() == "SINE":
            self.filter_type = "SINE"
        elif filter_type.upper() == "EXP":
            self.filter_type = "EXP"
        elif filter_type.upper() == "DEXP":
            self.filter_type = "DEXP"
        elif filter_type.upper() == "DELTA":
            self.filter_type = "DELTA"
        else:
            raise ValueError("Did not recognize filter type. Should be 'SINE', 'EXP', 'GAUSS'")
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
        else:
            raise ValueError("Did not recognize nonlin type. Should be 'CUBIC' or 'EXP'")
        self.predictors = lr_inputs
        self.fit_params = fit_params

    @property
    def lr_coef(self):
        return self.fit_params[:self.predictors.shape[1]]

    @property
    def filter_params(self):
        return self.fit_params[self.predictors.shape[1]:]

    @property
    def filter_function(self):
        if self.filter_type == "SINE":
            return sin_filter
        elif self.filter_type == "EXP":
            return exp_filter
        elif self.filter_type == "DEXP":
            return dexp_filter
        elif self.filter_type == "DELTA":
            return delta_filter
        else:
            return None

    def predict(self):
        """
        The predicts the output of this modeling step using the original inputs
        Returns:
            The predicted timeseries
        """
        if self.filter_type is None:
            lr_res = np.dot(self.predictors, self.lr_coef)
            filt_res = lr_res
        else:
            fit_function = make_full_fit_fun(self.filter_function)
            filt_res = fit_function(self.predictors, *self.fit_params)
        if self.nonlin_type is None:
            return filt_res
        else:
            return self.nonlin_function(filt_res, *self.nonlin_params)


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
    """
    Parametrization of a cubic nonlinearity applied to x
    """
    return a*(x**3) + b*(x**2) + c*x + d


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


def dexp_f(frames, s1, t1, t2):
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

    # fit of Rh6 units from trigeminal inputs
    tg_out = np.hstack((standardize(tg_on_prediction[:, None]), standardize(tg_off_prediction[:, None])))
    response_names = ["Fast_ON", "Slow_ON", "Fast_OFF", "Slow_OFF", "Delayed_OFF"]
    rh6_dict = {}
    # ff = make_full_fit_fun(dexp_filter)
    # p0, bounds = get_fit_initials("DEXP", tg_out.shape[1])
    fig, ax = pl.subplots()
    filter_time = np.arange(11) / -5.0
    for i in range(region_results["Rh6"].full_averages.shape[1]):
        output = standardize(region_results["Rh6"].full_averages[:, i])
        # lr, f_params = predict_response_w_filter(tg_out, output)
        resid_fun, p0 = make_dexp_residual_function(tg_out, output, 11)
        params = least_squares(resid_fun, p0).x
        # params = curve_fit(ff, tg_out, output, p0=p0, bounds=bounds)[0]
        # print("Type {0} coefficients: {1}".format(i, lr.coef_))
        print("Type {0} coefficients: {1}".format(i, params[-tg_out.shape[1]:]))
        f = dexp_f(np.arange(11), *params[:-2])
        ax.plot(filter_time, f)
        # filtered_out = bp_gauss_filter_fit_function(lr.predict(tg_out), *f_params)
        # filtered_out = ff(tg_out, *params)
        lr_sum = np.sum(tg_out * params[-tg_out.shape[1]:][None, :], 1)
        filtered_out = np.convolve(lr_sum, f)[:tg_on_prediction.size]
        nl_params = curve_fit(cubic_nonlin, filtered_out[f_s:], output[f_s:])[0]
        prediction = cubic_nonlin(filtered_out, *nl_params)
        print("Type {0} R2: {1}".format(i, r2(prediction[f_s:], output[f_s:])))
        print("Type {0} FVU: {1}".format(i, fvu(prediction[f_s:], output[f_s:])))
        # test contribution of tg components
        for j in range(tg_out.shape[1]):
            comp = tg_out[:, j]
            lrs = comp * params[-tg_out.shape[1]:][j]
            fout = np.convolve(lrs, f)[:tg_off_prediction.size]
            nlp = curve_fit(cubic_nonlin, fout[f_s:], output[f_s:])[0]
            red_pred = cubic_nonlin(fout, *nlp)
            print("Type {0} with tg component {1} R2: {2}".format(i, j, r2(red_pred[f_s:], output[f_s:])))
