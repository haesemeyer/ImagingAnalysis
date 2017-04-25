# script to analyze already segmented white noise stimulation experiments
from mh_2P import *
import pickle
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from scipy.signal import deconvolve
import sys
import pandas
sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')
import mhba_basic as mb


def repeat_aligned(data: np.ndarray):
    """
    For a given dataset returns a matrix with n-repeat rows containing in each row one experimental repeat
    """
    global s_pre, s_stim, s_post, interp_freq, n_repeats
    return data.reshape((n_repeats, (s_pre + s_stim + s_post)*interp_freq))


def stim_only(data: np.ndarray, start_offset=0):
    """
    For a given dataset returns a version that only contains data during stimulus periods optionally with an offset
    Args:
        data: The timeseries data for which to return stimulus periods
        start_offset: The time [s] within each stimulus period from which onward data should be returned

    Returns:
        Data within the stimulation period after the specified offset
    """
    global s_pre, s_stim, s_post, interp_freq
    return repeat_aligned(data)[:, (s_pre+start_offset)*interp_freq:(s_pre+s_stim)*interp_freq+1].ravel()


def cross_correlate(data1: np.ndarray, data2: np.ndarray, maxlag=1):
    def norm(data):
        data -= np.mean(data)
        n = np.linalg.norm(data)
        if n > 0:
            return data / np.linalg.norm(data)
        else:
            return data
    global interp_freq
    cc = np.correlate(norm(data1.copy()), norm(data2.copy()), mode="full")
    mp = cc.size//2
    start = mp - maxlag*interp_freq
    end = mp + maxlag*interp_freq
    return cc[start:end+1]


def plot_unit_kernel(g, maxlag=2, deconvolute=True):
    global laser_currents, s_pre, s_stim, s_post, interp_freq
    if deconvolute:
        dc_act = np.zeros_like(g.interp_data)
        deconv = deconvolve(g.interp_data, TailData.CaKernel(0.4, interp_freq))[0]
        dc_act[:deconv.size] = deconv
        act_rep_aligned = repeat_aligned(dc_act)  # repeat_aligned(g.interp_data)
    else:
        act_rep_aligned = repeat_aligned(g.interp_data)
    laser_rep_aligned = repeat_aligned(laser_currents)
    # for our cross-correlation kernels we only use data from 10s after stim start
    start = (s_pre + 10) * interp_freq
    cc_len = 2*maxlag*interp_freq + 1
    cross_corrs = np.zeros((act_rep_aligned.shape[0], cc_len))
    for i, (act, laser) in enumerate(zip(act_rep_aligned, laser_rep_aligned)):
        cross_corrs[i, :] = cross_correlate(laser[start:], act[start:], maxlag)
    # plot trial average activity as well as kernel (both bootstrapped)
    kernel_time = np.linspace(-maxlag, maxlag, cc_len)
    trial_time = np.linspace(0, s_pre + s_stim + s_post, interp_freq * (s_pre + s_stim + s_post), endpoint=False)
    F0 = np.mean(act_rep_aligned[:, 10*interp_freq:s_pre*interp_freq], 1, keepdims=True)
    with sns.axes_style('whitegrid'):
        fig, (ax1, ax2, ax3) = pl.subplots(ncols=3)
        sns.tsplot((act_rep_aligned-F0) / F0, trial_time, ax=ax1, ci=95)
        ax1.set_xlim(s_pre-10, s_pre+10)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("dF/F0")
        ax1.set_title("Trial average cell activity")
        means, stds = pre_stim_differences(g)
        dframe = pandas.DataFrame({"AVG": means[:, 1]-means[:, 0], "STD": stds[:, 1]-stds[:, 0]})
        sns.barplot(data=dframe, ax=ax2)
        ax2.set_title("Pre-Stim differences in activity")
        sns.tsplot(cross_corrs, kernel_time, ax=ax3, ci=95, interpolate=False, err_style="ci_bars")
        pl.plot([-maxlag, maxlag], [0, 0], 'k--')
        ax3.set_xlabel("Lag [s]")
        ax3.set_ylabel("Cross-correlation r")
        ax3.set_title("Cell kernel")
        fig.tight_layout()


def computeKernel(g, nboot=500):
    pre = 2 * interp_freq
    post = interp_freq
    # find frame indices that are valid starts and ends for model derivation
    frames = np.arange(g.interp_data.size)
    start = (s_pre + 10) * interp_freq + pre
    end = (s_pre + s_stim) * interp_freq - post
    fr_rep_aligned = repeat_aligned(frames)[:, start:end]
    coefs = np.zeros((nboot, pre+post+1))
    # dc_act = deconvolve(g.interp_data, TailData.CaKernel(0.4, interp_freq))[0]
    for i in range(nboot):
        chosen_trials = np.random.choice(np.arange(n_repeats), n_repeats)
        frames_taken = np.zeros_like(fr_rep_aligned)
        for j, ct in enumerate(chosen_trials):
            frames_taken[j, :] = fr_rep_aligned[ct, :]
        val_frames = frames_taken.ravel()
        ix = mb.IndexingMatrix(val_frames, pre, post, g.interp_data.size)[0]
        laser_in = laser_currents[ix]
        act_out = g.interp_data[val_frames]
        assert act_out.size == laser_in.shape[0]
        eln = ElasticNet(l1_ratio=0.25, alpha=0.05)
        eln.fit(laser_in, act_out)
        coefs[i, :] = eln.coef_
    return coefs


def computeKernel2(g, nboot=500, alpha=1e-6):
    import statsmodels.api as sm
    pre = 3 * interp_freq
    post = interp_freq
    # find frame indices that are valid starts and ends for model derivation
    frames = np.arange(g.interp_data.size)
    start = (s_pre + 5) * interp_freq + pre
    end = (s_pre + s_stim) * interp_freq - post
    fr_rep_aligned = repeat_aligned(frames)[:, start:end]
    if nboot < 2:
        val_frames = fr_rep_aligned.ravel()
        ix = mb.IndexingMatrix(val_frames, pre, post, g.interp_data.size)[0]
        laser_in = laser_currents[ix]
        exog = sm.add_constant(laser_in)
        act_out = g.interp_data[val_frames]
        assert act_out.size == laser_in.shape[0]
        poisson_model = sm.GLM(act_out, exog, family=sm.families.Poisson())
        return poisson_model.fit()
    else:
        coefs = np.zeros((nboot, pre+post+1))
        icepts = np.zeros(nboot)
        # dc_act = deconvolve(g.interp_data, TailData.CaKernel(0.4, interp_freq))[0]
        for i in range(nboot):
            chosen_trials = np.random.choice(np.arange(n_repeats), n_repeats)
            frames_taken = np.zeros_like(fr_rep_aligned)
            for j, ct in enumerate(chosen_trials):
                frames_taken[j, :] = fr_rep_aligned[ct, :]
            val_frames = frames_taken.ravel()
            ix = mb.IndexingMatrix(val_frames, pre, post, g.interp_data.size)[0]
            laser_in = laser_currents[ix]
            exog = sm.add_constant(laser_in)
            act_out = g.interp_data[val_frames]
            assert act_out.size == laser_in.shape[0]
            poisson_model = sm.GLM(act_out, exog, family=sm.families.Poisson())
            alphas = np.zeros(exog.shape[1])
            alphas[1:] = alpha  # do not regularize constant term!
            poisson_results = poisson_model.fit()
            coefs[i, :] = poisson_results.params[1:]
            icepts[i] = poisson_results.params[0]
        return coefs, icepts


def computeKernel3(g, alpha=1e-6):
    import statsmodels.api as sm
    pre = 2 * interp_freq
    post = interp_freq
    # find frame indices that are valid starts and ends for model derivation
    frames = np.arange(g.interp_data.size)
    start = (s_pre + 5) * interp_freq + pre
    end = (s_pre + s_stim) * interp_freq - post
    fr_rep_aligned = repeat_aligned(frames)[:, start:end]
    coefs = np.zeros((n_repeats, pre+post+1))
    icepts = np.zeros(n_repeats)
    # dc_act = deconvolve(g.interp_data, TailData.CaKernel(0.4, interp_freq))[0]
    for i in range(n_repeats):
        val_frames = fr_rep_aligned[i, :]
        ix = mb.IndexingMatrix(val_frames, pre, post, g.interp_data.size)[0]
        laser_in = laser_currents[ix]
        exog = sm.add_constant(laser_in)
        act_out = g.interp_data[val_frames]
        assert act_out.size == laser_in.shape[0]
        poisson_model = sm.GLM(act_out, exog, family=sm.families.Poisson())
        alphas = np.zeros(exog.shape[1])
        alphas[1:] = alpha  # do not regularize constant term!
        poisson_results = poisson_model.fit_regularized(alpha=alphas, L1_wt=0.5)
        coefs[i, :] = poisson_results.params[1:]
        icepts[i] = poisson_results.params[0]
    return coefs, icepts


def pre_stim_differences(g):
    """
    For the given cell graph computes per-trial pre and stim means and standard deviations (pre: 10s before phase switch
    stim: 10s after phase switch)

    Returns:
        [0]: A n_trial x 2 array of the pre- and stim activity means
        [1]: A n_trial x 2 array of the pre- and stim activity standard deviations
    """
    global s_pre, s_stim, interp_freq
    act_rep_aligned = repeat_aligned(g.interp_data)
    F0 = np.mean(act_rep_aligned[:, 10 * interp_freq:s_pre * interp_freq], 1, keepdims=True)
    dff = (act_rep_aligned - F0) / F0
    pre = dff[:, (s_pre - 10)*interp_freq:s_pre*interp_freq]
    post = dff[:, s_pre*interp_freq:(s_pre + 10)*interp_freq]
    means = np.hstack((np.mean(pre, 1, keepdims=True), np.mean(post, 1, keepdims=True)))
    stds = np.hstack((np.std(pre, 1, keepdims=True), np.std(post, 1, keepdims=True)))
    return means, stds


def paired_bstrap_p(x, y=None, kind="<>", nboot=5000):
    """
    Performs boot-strap test on averages of paired observations
    Args:
        x: If y is None, compare the contents of x with 0, otherwise test for y-x
        y: Observations paired with x
        kind: "<>" for two-sided, ">" for y greater x, "<" for y smaller x
        nboot: Number of bootstrap samples to use

    Returns:
        The p-value of the test
    """
    if y is None:
        diff = x
    else:
        diff = y-x
    test_diff = np.mean(diff)
    # return for obvious cases
    if test_diff > 0 and kind == "<":
        return 1
    elif test_diff < 0 and kind == ">":
        return 1
    bg_diff = np.zeros(5000)
    for i in range(nboot):
        bd = diff[np.random.choice(np.arange(diff.size), diff.size)]
        bg_diff[i] = np.mean(bd)
    if kind == "<" or test_diff < 0:
        p = np.sum(bg_diff >= 0) / nboot
    elif kind == ">" or test_diff > 0:
        p = np.sum(bg_diff <= 0) / nboot
    else:
        p = 1
    if kind == "<>":
        return p * 2
    return p

# laser current measurement constants
lc_offset = -0.014718754910138
lc_scale = -1.985632024505160e+02

if __name__ == "__main__":
    sns.reset_orig()
    n_repeats = int(input("Please enter the number of repeats:"))
    s_pre = int(input("Please enter the number of pre-seconds:"))
    s_stim = int(input("Please enter the number of stimulus-seconds:"))
    try:
        s_post = int(input("Please enter the number of post-seconds or press enter for 0:"))
    except ValueError:
        print("Assuming default value of 0")
        s_post = 0
    total_seconds = (s_pre + s_stim + s_post)*n_repeats
    t_per_frame = float(input("Please enter the duration of each frame in seconds:"))

    interp_freq = 5  # set final frequency low to start

    fname = UiGetFile([('Graph files', '.graph')])
    file = open(fname, "rb")
    try:
        graph_list = pickle.load(file)
    finally:
        file.close()
    ext_start = fname.find(".graph")
    laser_file = fname[:ext_start-2] + ".laser"
    laser_currents = (np.genfromtxt(laser_file)-lc_offset) * lc_scale
    frame_times = np.arange(graph_list[0].RawTimeseries.size) * t_per_frame
    # first interpolate to 10Hz, then bin down to our interp_frew
    interp_times = np.linspace(0, total_seconds, total_seconds*10, endpoint=False)
    ipol = lambda y: np.interp(interp_times, frame_times, y[:frame_times.size])
    for g in graph_list:
        g.interp_data = np.add.reduceat(ipol(g.RawTimeseries), np.arange(0, interp_times.size, 10//interp_freq))
    laser_currents = np.add.reduceat(laser_currents, np.arange(0, laser_currents.size, 20//interp_freq)) / (20//interp_freq)
    laser_currents = laser_currents[:g.interp_data.size]

    # convert laser currents to power levels based on collimator output measurements
    currents = np.arange(500, 1600, 100)
    measured_powers = np.array([18, 46, 71, 93, 115, 137, 155, 173, 192, 210, 229])
    lr = LinearRegression()
    lr.fit(currents[:, None], measured_powers[:, None])
    laser_currents = lr.predict(laser_currents[:, None]).ravel()
    laser_currents[laser_currents < 0] = 0
    # convert to temperature representation
    laser_currents = CaConvolve(laser_currents, 0.891, interp_freq)

    diffs = np.array([pre_stim_differences(g) for g in graph_list])
    p = np.array([paired_bstrap_p(d[0][:, 0], d[0][:, 1]) for d in diffs])

    # load and bin down bout start trace - use non-convolved
    tdd = TailDataDict(graph_list[0].CaTimeConstant)
    taildata = tdd[graph_list[0].SourceFile]
    bstarts = taildata.starting
    times = taildata.frameTime
    i_t = np.linspace(0, total_seconds, total_seconds*interp_freq + 1, endpoint=True)
    digitized = np.digitize(times, i_t)
    binned_bstarts = np.array([bstarts[digitized == i].sum() for i in range(1, i_t.size)])
