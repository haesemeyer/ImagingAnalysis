# toy simulation using neuronal model to simulate gradient behavior
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import pickle
import matplotlib as mpl
from typing import Dict
from sensMotorModel import ModelResult, run_model

frame_rate = 5
avg_bout_freq = 0.92  # from real gradient data
p_bout = 0.92 / frame_rate
std_bout_freq = 0.43
sd_p_bout = 0.43 / frame_rate
avg_bout_len_ms = 194  # from real gradient data
avg_blen_frames = np.round(avg_bout_len_ms / 1000 * frame_rate)
# behavioral distributions from real gradient data used for swims
# all assumed to be independent of temperature
sw_da_mean = -1
sw_da_sd = 41
sw_disp_k = 2.63  # = alpha of gamma
sw_disp_theta = 1/0.138  # = 1/beta of gamma

# flick related parameters are not based on data but just meant to reflect a steep in-place turn
fl_da_mean = 45
fl_da_sd = 10
fl_disp_mean = 0
fl_disp_sd = 0

# arena parameters
arena_radius_px = 55 * 9


class ModelSimulation:
    def __init__(self, model_results):
        self.mr = model_results
        self.__rn_cash = np.random.randn(1000)
        self.__rnc_count = -1
        self.__g_cash = np.random.gamma(sw_disp_k, sw_disp_theta, 1000)
        self.__gc_count = -1
        self.__u_cash = np.random.rand(1000)
        self.__uc_count = -1

    def __reset_cashes(self, size):
        self.__rn_cash = np.random.randn(size)
        self.__rnc_count = -1
        self.__g_cash = np.random.gamma(sw_disp_k, sw_disp_theta, size)
        self.__gc_count = -1
        self.__u_cash = np.random.rand(size)
        self.__uc_count = -1

    def __next_normal(self):
        self.__rnc_count += 1
        if self.__rnc_count < self.__rn_cash.size:
            return self.__rn_cash[self.__rnc_count]
        else:
            # grow cash
            self.__rn_cash = np.random.randn(self.__rn_cash.size * 2)
            self.__rnc_count = 0
            return self.__rn_cash[self.__rnc_count]

    def __next_gamma(self):
        self.__gc_count += 1
        if self.__gc_count < self.__g_cash.size:
            return self.__g_cash[self.__gc_count]
        else:
            # grow cash
            self.__g_cash = np.random.gamma(sw_disp_k, sw_disp_theta, self.__g_cash.size*2)
            self.__gc_count = 0
            return self.__g_cash[self.__gc_count]

    def __next_uni(self):
        self.__uc_count += 1
        if self.__uc_count < self.__u_cash.size:
            return self.__u_cash[self.__uc_count]
        else:
            # grow cash
            self.__u_cash = np.random.rand(self.__u_cash.size * 2)
            self.__uc_count = 0
            return self.__u_cash[self.__uc_count]

    @staticmethod
    def temp(r, exp=1):
        return ((r / arena_radius_px)**exp) * 10 + 22

    def select_bout(self, p_swim, p_flick):
        """
        Given a set of probabilities performs a given bout or not
        Args:
            p_swim: The probability to initiate a swim
            p_flick: The probability to initiate a flick

        Returns:
            [0]: The displacement in pixels
            [1]: The delta angle change
            [2]: The number of frames the selected move takes
        """
        dec = self.__next_uni()
        if dec < p_swim:
            da = self.__next_normal()*sw_da_sd + sw_da_mean
            disp = self.__next_gamma()
            return disp, da, int(avg_blen_frames)
        elif dec < p_swim + p_flick:
            left = self.__next_uni() < 0.5
            if left:
                return 0, self.__next_normal()*fl_da_sd - fl_da_mean, int(avg_blen_frames)
            else:
                return 0, self.__next_normal() * fl_da_sd + fl_da_mean, int(avg_blen_frames)
        else:
            return 0, 0, 1

    @classmethod
    def implement_bout(cls, xstart, ystart, angstart, disp, da, nframes):
        """
        Implements a bout from the give starting conditions returning a trajectory of the bout
        Args:
            xstart: The starting x-position
            ystart: The starting y-position
            angstart: The starting angle
            disp: The bout displacement
            da: The bout angle change
            nframes: The number of frames in the bout

        Returns:
            [0]: Bout trajectory x-coordinate
            [1]: Bout trajectory y-coordinates
            [3]: Bout trajectory heading-angles
            [4]: Bout trajectory temperatures
        """
        ang_new = angstart + da
        dx = disp * np.cos(np.deg2rad(ang_new))
        dy = disp * np.sin(np.deg2rad(ang_new))
        # reflect bout if this would take us out of the arena
        if np.sqrt((xstart+dx)**2 + (ystart+dy)**2) > arena_radius_px:
            ang_new = ang_new - 180
            dx = disp * np.cos(np.deg2rad(ang_new))
            dy = disp * np.sin(np.deg2rad(ang_new))
        traj_x = np.zeros(nframes)
        traj_y = np.zeros_like(traj_x)
        traj_a = np.zeros_like(traj_x)
        for i in range(traj_x.size):
            traj_x[i] = xstart + dx * (i+1) / traj_x.size
            traj_y[i] = ystart + dy * (i+1) / traj_y.size
            traj_a[i] = ang_new
        traj_t = cls.temp(np.sqrt(traj_x ** 2 + traj_y ** 2))
        return traj_x, traj_y, traj_a, traj_t

    def do_simulation(self, n_steps, n_burn=20, control=False, sw_eq_fl=False):
        """
        Performs one simulation run
        Args:
            n_steps: The number of steps to perform in the simulation
            n_burn: Number of burn-in steps to avoid initial transient when evaluating model
            control: If True, bout frequency won't depend on temperature
            sw_eq_fl: If True the model will only predict average bout rate but swimming and flicking will occur with
                equal probablity

        Returns:
            [0]: Array of x-positions at each simulation step
            [1]: Array of y-positions at each simulation step
            [2]: Array of temperatures at each simulation step
            [3]: Array of heading angles at each simulation step
        """
        xp = np.zeros(n_steps + n_burn)
        yp = np.zeros_like(xp)
        temp = np.zeros_like(xp)
        angs = np.zeros_like(xp)
        curr_time = n_burn
        while curr_time < n_steps + n_burn:
            if control:
                psw, pfl = p_bout / 2, p_bout / 2
            else:
                model_in = temp[:curr_time - 1]
                if model_in.size > 100:
                    model_in = model_in[-100:]
                model_in = (model_in - m_in) / s_in
                psw, pfl = run_model(model_in, self.mr)[:2]
                if sw_eq_fl:
                    psw = (psw + pfl) / 2
                    pfl = psw
                psw = (psw[-1] * sd_p_bout + p_bout) / 2
                pfl = (pfl[-1] * sd_p_bout + p_bout) / 2
            ds, da, n = self.select_bout(psw, pfl)
            tx, ty, ta, tt = self.implement_bout(xp[curr_time-1], yp[curr_time-1], angs[curr_time-1], ds, da, n)
            if curr_time + n >= n_steps + n_burn:
                n = n_steps + n_burn - curr_time
            xp[curr_time:curr_time+n] = tx[:n]
            yp[curr_time:curr_time+n] = ty[:n]
            temp[curr_time:curr_time+n] = tt[:n]
            angs[curr_time:curr_time+n] = ta[:n]
            curr_time += n
            if curr_time % 1000 == 0:
                print('\b' * len(str(curr_time)) + str(curr_time), end='', flush=True)
        return xp, yp, temp, angs


if __name__ == "__main__":
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    # load model results
    model_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/model_170713.hdf5', 'r')
    model_results = pickle.loads(np.array(model_file["model_results"]))  # type: Dict[str, ModelResult]
    stim_in = np.array(model_file["stim_in"])
    m_in = np.array(model_file["m_in"])
    s_in = np.array(model_file["s_in"])
    model_file.close()
    sim = ModelSimulation(model_results)
    n_steps = 1000000
    res_full = sim.do_simulation(n_steps)
    print("Full simulation completed")
    res_cont = sim.do_simulation(n_steps, control=True)
    print("Control simulation completed")
    res_eq = sim.do_simulation(n_steps, sw_eq_fl=True)
    print("Swim=Flick simulation completed")
    r_full = np.sqrt(res_full[0] ** 2 + res_full[1] ** 2)
    r_cont = np.sqrt(res_cont[0] ** 2 + res_cont[1] ** 2)
    r_eq = np.sqrt(res_eq[0] ** 2 + res_eq[1] ** 2)

    pl.figure()
    sns.distplot(r_cont, label="Control")
    sns.distplot(r_eq, label="Swim=Flick")
    sns.distplot(r_full, label="Full")
    pl.legend()
    sns.despine()
