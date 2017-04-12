# script to analyze already segmented detailed characeterization experiments
from mh_2P import CellGraph, UiGetFile, CaConvolve, MotorContainer, DetailCharExperiment
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from os import path
import sys
sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/OASIS')
import functions as oa_func

# laser current measurement constants
lc_offset = -0.014718754910138
lc_scale = -1.985632024505160e+02


def oasis_g_from_t_half(decay_halftime: float, time_per_frame: float):
    """
    
    Args:
        decay_halftime: The decay halftime of the calcium indicator
        time_per_frame: The acquisition time of each frame

    Returns:
        The constant g of an AR(1) process corresponding to the parameters
    """
    # compute constant t_d of exponential kernel from half-time
    t_d = decay_halftime / np.log(2)
    freq = 1 / time_per_frame
    return np.exp(-1 / (t_d * freq))

if __name__ == "__main__":
    sns.reset_orig()
    n_repeats = int(input("Please enter the number of repeats:"))
    s_pre = 0
    s_stim = 135
    s_post = 0
    total_seconds = (s_pre + s_stim + s_post)*n_repeats
    t_per_frame = float(input("Please enter the duration of each frame in seconds:"))

    interp_freq = 5

    fnames = UiGetFile([('Graph files', '.graph')], multiple=True)
    graph_list = []
    for fname in fnames:
        file = open(fname, "rb")
        try:
            graph_list += pickle.load(file)
        finally:
            file.close()
    sourceFiles = [(g.SourceFile, t_per_frame) for g in graph_list]
    # load laser file of first experiment (they should all be the same anyways...)
    ext_start = sourceFiles[0][0].find(".tif")
    laser_file = sourceFiles[0][0][:ext_start-2] + ".laser"
    laser_currents = (np.genfromtxt(laser_file)-lc_offset) * lc_scale
    frame_times = np.arange(graph_list[0].RawTimeseries.size) * t_per_frame
    # interpolate to our interp-frequency after filtering timeseries using oasis
    gval = oasis_g_from_t_half(0.4, t_per_frame)
    interp_times = np.linspace(0, total_seconds, total_seconds*interp_freq, endpoint=False)
    ipol = lambda y: np.interp(interp_times, frame_times, y[:frame_times.size])
    for g in graph_list:
        g.interp_data = ipol(oa_func.deconvolve(g.RawTimeseries.astype(float), g=[gval], penalty=1)[0])
    # bin down the laser currents to our interpolation frequency
    laser_currents = np.add.reduceat(laser_currents, np.arange(0, laser_currents.size, 20//interp_freq))
    laser_currents /= (20//interp_freq)
    laser_currents = laser_currents[:graph_list[0].interp_data.size]

    # convert laser currents to power levels based on collimator output measurements
    currents = np.arange(500, 1600, 100)
    measured_powers = np.array([18, 46, 71, 93, 115, 137, 155, 173, 192, 210, 229])
    lr = LinearRegression()
    lr.fit(currents[:, None], measured_powers[:, None])
    laser_currents = lr.predict(laser_currents[:, None]).ravel()
    laser_currents[laser_currents < 0] = 0
    # convert to temperature representation
    laser_currents = CaConvolve(laser_currents, 0.891, interp_freq)

    # create a motor container
    i_time = np.linspace(0, g.interp_data.size / 5, g.interp_data.size + 1)
    mc_all_raw = MotorContainer(sourceFiles, i_time, 0)
    mc_all = MotorContainer(sourceFiles, i_time, 0.4, tdd=mc_all_raw.tdd)

    # plot average response with time of tap drawn in
    motor_by_trial = mc_all_raw.avg_motor_output.reshape((n_repeats, mc_all_raw[0].size // n_repeats))
    mean_res = np.mean(motor_by_trial, 0)
    time = np.arange(motor_by_trial.shape[1]) / interp_freq
    fig, ax = pl.subplots()
    sns.tsplot(data=motor_by_trial, time=time, ax=ax)
    ax.plot([130, 130], [np.min(mean_res), np.max(mean_res)], 'k--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Swim probability')
    ax.set_xlim(0, 135)
    sns.despine(fig, ax)
    # plot predicted trial temperature
    fig, ax = pl.subplots()
    ax.plot(np.arange(laser_currents.size) / interp_freq, laser_currents, 'r')
    ax.annotate("", xy=(130, 0), xytext=(130, 30), arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 135)
    ax.set_ylabel('Temperature [AU]')
    sns.despine(fig, ax)

    # gather all activity and all convolved motor
    all_activity = np.vstack([g.interp_data for g in graph_list])
    all_motor = np.vstack([mc for mc in mc_all])
    assert all_activity.shape[0] == all_motor.shape[0]
    assert all_activity.shape[1] == all_motor.shape[1]

    # create experiment class
    g_info = [(g.SourceFile, g.V) for g in graph_list]
    data = DetailCharExperiment(all_activity, all_motor, n_repeats, t_per_frame, g_info)
    # save data to file
    z_start = graph_list[0].SourceFile.find("_Z_")
    save_name = graph_list[0].SourceFile[:z_start] + ".pickle"
    if path.exists(save_name):
        raise FileExistsError("Save file {0} already exists. Aborting".format(save_name))
    f = open(save_name, "wb")
    try:
        pickle.dump(data, f)
    finally:
        f.close()
