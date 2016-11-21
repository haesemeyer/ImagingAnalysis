from collections import deque
import numpy as np
from PIL import Image

from collections import Counter

import matplotlib.pyplot as pl
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import lfilter
from scipy.interpolate import interp1d

from numba import jit
import numba
from numpy import std, zeros

import cv2

import sys

import warnings

try:
    sys.path.append('C:/Users/mhaesemeyer/Documents/Python Scripts/BehaviorAnalysis')
    import mhba_basic as mb
except ImportError:
    sys.path.append('/Users/mhaesemeyer/Documents/Python/BehaviorAnalysis')
    import mhba_basic as mb

try:
    import Tkinter
    import tkFileDialog
except ImportError:
    import tkinter as Tkinter
    import tkinter.filedialog as tkFileDialog


class GraphBase:
    """
    Base class for pixel based graphs that represent segmented regions in timeseries stacks
    """
    def __init__(self):
        self.V = []
        self.ID = None
        self.RawTimeseries = np.array([])
        self.shuff_ts = []
        self.FramesPre = None
        self.FramesStim = None
        self.FramesPost = None

    @property
    def NPixels(self):
        return len(self.V)

    @property
    def MinX(self):
        return min(list(zip(*self.V))[0])

    @property
    def MinY(self):
        return min(list(zip(*self.V))[1])

    @property
    def MaxX(self):
        return max(list(zip(*self.V))[0])

    @property
    def MaxY(self):
        return max(list(zip(*self.V))[1])

    @property
    def CenterOfMass(self):
        x, y, gen = list(zip(*self.V))
        return np.mean(x), np.mean(y)

    def ComputeGraphRotations(self, nrotations):
        min_shuff = self.FramesPre // 3
        max_shuff = self.RawTimeseries.size - self.FramesPre // 3
        shuff_ts = np.zeros((nrotations, self.RawTimeseries.size))
        rolls = np.random.randint(min_shuff, max_shuff, size=nrotations)
        for i in range(nrotations):
            shuff_ts[i, :] = np.roll(self.RawTimeseries, rolls[i])
        self.shuff_ts = shuff_ts

    @staticmethod
    def ZScore(trace, axis=-1):
        if trace.ndim == 2 and axis != -1:
            m = np.mean(trace, axis, keepdims=True)
            s = np.std(trace, axis, keepdims=True)
            return (trace - m) / s
        else:
            return (trace - np.mean(trace)) / np.std(trace)

    @staticmethod
    def FourierMax(timeseries, axis=-1):
        """
        Returns the maximal magnitude in the fourier spectrum of the zscored timeseries
        """
        return np.max(np.absolute(np.fft.rfft(GraphBase.ZScore(timeseries, axis), axis=axis)), axis=axis)

    @staticmethod
    def FourierEntropy(timeseries, axis=-1):
        """
        Computes the entropy of the fourier spectrum of the zscored timeseries
        """
        magnitudes = np.absolute(np.fft.rfft(GraphBase.ZScore(timeseries, axis), axis=axis))
        sm = np.sum(magnitudes, axis=axis, keepdims=True)
        magnitudes /= sm
        e = magnitudes * np.log2(magnitudes)
        return -1 * np.nansum(e, axis=axis)

    def TimeseriesShuffles(self, nshuffles):
        """
        Returns a shuffle matrix of the graphs RawTimeseries
        """
        ts_shuff = np.random.choice(self.RawTimeseries, self.RawTimeseries.size * nshuffles)
        ts_shuff = np.reshape(ts_shuff, (nshuffles, self.RawTimeseries.size))
        return ts_shuff

    def FourierMaxShuffles(self, nshuffles):
        """
        Performs shuffles in timeseries and returns the distribution of associated fourier maxima
        """
        ts_shuff = self.TimeseriesShuffles(nshuffles)
        return self.FourierMax(ts_shuff, axis=1)

    def FourierMaxPValue(self, nshuffles):
        """
        Determines the level of significance (p-value) of the fourier maximum in the timeseries using
        random shuffles as a background distribution
        """
        max_real = self.FourierMax(self.RawTimeseries)
        bg_distrib = self.FourierMaxShuffles(nshuffles)
        return np.sum(bg_distrib >= max_real) / nshuffles

    def FourierEntropyShuffles(self, nshuffles):
        """
        Performs shuffles in timeseries and returns the distribution of associated fourier entropies
        """
        ts_shuff = self.TimeseriesShuffles(nshuffles)
        return self.FourierEntropy(ts_shuff, axis=1)

    def FourierEntropyPValue(self, nshuffles):
        """
        Calculates level of significance of the entropy of the real fourier spectrum being
        smaller (i.e. more peaky) than the entropy of shuffles
        """
        e_real = self.FourierEntropy(self.RawTimeseries)
        bg_distrib = self.FourierEntropyShuffles(nshuffles)
        return np.sum(bg_distrib <= e_real) / nshuffles


class NucGraph(GraphBase):
    """
    Represents a nuclear segmentation that was performed by an outside program which
    encoded different nuclei as intensities in an image
    """

    def __init__(self, id, timeseries):
        super().__init__()
        self.V = []
        self.ID = id
        self.RawTimeseries = timeseries
        self.gcv = []  # the greyscale values of all graph pixels
        self.shuff_ts = []

    @staticmethod
    def NuclearConnComp(stack, segmentImage, predicate):
        """
        Uses a presegmentation saved as an image (such as obtained from cell profiler) to build
        a list of graphs where each graph corresponds to one connected component of one intensity
        value in segmentImage
        Args:
            stack: The original stack to add time-series data to the graph
            segmentImage: An image with nuclear segmentations encoded by intensity

        Returns: A list of nucleus graphs that segment segmentImage

        """
        def BFS(sourceX, sourceY, color):
            """
            Performs breadth first search on sumImage
            given (sourceX,sourceY) as starting pixel
            coloring all visited pixels in color
            """
            if stack is not None:
                pg = NucGraph(color, np.zeros(stack.shape[0], dtype=np.float32))
            else:
                pg = NucGraph(color, 0)
            Q = deque()
            Q.append((sourceX, sourceY, 0))
            visited[sourceX, sourceY] = color  # mark source as visited
            while len(Q) > 0:
                v = Q.popleft()
                x = v[0]
                y = v[1]
                pg.V.append(v)  # add current vertex to pixel graph
                if sumstack is not None:
                    pg.gcv.append(sumstack[x, y])
                    pg.RawTimeseries += stack[:, x, y]  # add current pixel's timeseries
                # add non-visited neighborhood to queue - 8-connected neighborhood
                for xn in range(x - 1, x + 2):  # x+1 inclusive!
                    for yn in range(y - 1, y + 2):
                        if xn < 0 or yn < 0 or xn >= segmentImage.shape[0] or yn >= segmentImage.shape[1] or visited[xn, yn] != 0:
                            # outside image dimensions or already visited
                            continue
                        if not predicate(xn, yn):
                            continue
                        # pixel should be added if it has the same value as the source
                        if segmentImage[xn, yn] == color:
                            Q.append((xn, yn, v[2] + 1))  # add non-visited above threshold neighbor
                            visited[xn, yn] = color  # mark as visited
            return pg
        if stack is not None:
            sumstack = np.sum(stack, 0)
        else:
            sumstack = None
        conn_comps = []  # list of nucleus graphs
        visited = np.zeros_like(segmentImage)
        nr, nc = segmentImage.shape
        for x in range(nr):
            for y in range(nc):
                if visited[x, y] == 0 and segmentImage[x, y] != 0 and predicate(x, y):
                    conn_comps.append(BFS(x, y, segmentImage[x, y]))
        return conn_comps, visited


class CorrelationGraph(GraphBase):

    def __init__(self, id, timeseries):
        super().__init__()
        self.V = []  # list of vertices
        self.ID = id  # id of this graph for easy component reference
        self.Timeseries = timeseries  # the summed pixel-timeseries of the graph

    @staticmethod
    def CorrelationConnComps(stack, im_ncorr, corr_thresh, predicate, norm8=True, limit=None, seed_limit=0):
        """
        Builds connected component graphs whereby components are
        determined based on pixel-timeseries correlations. Pixels
        with timeseries correlations >= corr_thresh are considered
        as sources for a breadth-first-search which will incor-
        porate new pixels into the graph if they are correlated
        with at least corr_thresh to the summed trace of the
        component. Every new pixel's timeseries will be added
        to the sum-trace of the component.
            stack: Image time-seried stack [t,x,y]
            im_ncorr: Image with same x,y dimensions as stack
              initialized to neighborhood correlations of each pixel
            corr_thresh: Threshold correlation for source consideration
              and pixel incorporation.
            norm8: Use 8 or 4 connected components
            limit: Optionally specify start and end of timeslice over
              which correlations should be computed
            seed_limit: For a pixel to be considered a valid seed, it's
              neighborhood correlation needs to exceed this value
            RETURNS:
                [0]: List of connected component graphs
                [1]: Image numerically identifying each pixel of each graph
        """
        def BFS(stack, thresh, visited, sourceX, sourceY, color, norm8, predicate):
            """
            Performs breadth first search on image
            given (sourceX,sourceY) as starting pixel
            coloring all visited pixels in color
            """
            nonlocal limit
            pg = CorrelationGraph(color, np.zeros(stack.shape[0], dtype=np.float32))
            Q = deque()
            Q.append((sourceX, sourceY, 0))
            visited[sourceX, sourceY] = color  # mark source as visited
            while len(Q) > 0:
                v = Q.popleft()
                x = v[0]
                y = v[1]
                pg.V.append(v)  # add current vertex to pixel graph
                pg.Timeseries = pg.Timeseries + stack[:, x, y]  # add current pixel's timeseries
                # add non-visited neighborhood to queue
                for xn in range(x-1, x+2):  # x+1 inclusive!
                    for yn in range(y-1, y+2):
                        # outside image dimensions or already visited
                        if xn < 0 or yn < 0 or xn >= stack.shape[1] or yn >= stack.shape[2] or visited[xn, yn]:
                            continue
                        if (not norm8) and xn != x and yn != y:
                            continue
                        if not predicate(xn, yn):
                            continue
                        # compute correlation of considered pixel's timeseries to full graphs timeseries
                        c = np.corrcoef(pg.Timeseries[limit[0]:limit[1]], stack[limit[0]:limit[1], xn, yn])[0, 1]
                        if c >= thresh:
                            Q.append((xn, yn, v[2]+1))  # add non-visited above threshold neighbor
                            visited[xn, yn] = color  # mark as visited
            return pg

        if im_ncorr.shape[0] != stack.shape[1] or im_ncorr.shape[1] != stack.shape[2]:
            raise ValueError("Stack width and height must be same as im_ncorr width and height")
        if limit is None:
            limit = (0, stack.shape[0])
        if len(limit) != 2:
            raise ValueError("If set limit must be a 2-element tuple or list of start and end slice (exclusive)")
        visited = np.zeros_like(im_ncorr, dtype=int)  # indicates visited pixels > 0
        conn_comps = []  # list of correlation graphs
        # at each iteration we find the pixel with the highest neighborhood correlation,
        # ignoring visited pixels, and use it as a source pixel for breadth first search
        curr_color = 1  # id counter of connected components
        while np.max(im_ncorr * (visited == 0)) > seed_limit:
            (x, y) = np.unravel_index(np.argmax(im_ncorr * (visited == 0)), im_ncorr.shape)
            conn_comps.append(BFS(stack, corr_thresh, visited, x, y, curr_color, norm8, predicate))
            curr_color += 1
        return conn_comps, visited
# class CorrelationGraph


class ImagingData:
    """
    Represents timeseries data from segmented regions across imaging planes and potentially experiments
    of the same structure
    """
    def __init__(self, imagingData, swimVigor):
        """
        Creates a new ImagingData class
        Args:
            imagingData: An n-units by n-frames sized matrix of timeseries data
            swimVigor:  An n-units by n-frames sized matrix of ca-convolved swim vigor at imaging frame rate
        """
        if imagingData.shape != swimVigor.shape:
            raise ValueError("Imaging data and swim vigor need to have the same dimensions")
        self.RawData = imagingData
        self.Vigor = swimVigor.astype(np.float32)

    @property
    def NFrames(self):
        return self.RawData.shape[1]

    @property
    def NUnits(self):
        return self.RawData.shape[0]

    def motorCorrelation(self, nrolls=0, rollMin=0, rollMax=None):
        """
        Computes the correlation between imaging responses and motor activity for each unit (row in data matrices)
        Args:
            nrolls: Optionally also computes mean and standard deviation of shuffle correlations
            rollMin: The minimum shuffle distance
            rollMax: The maximum shuffle distance. If None will be set to the number of columns, i.e. timepoints

        Returns: For each unit the correlation of its activity to the motor output and optionally shuffled mean
                 and standard deviation

        """
        return self.correlate(self.RawData, self.Vigor, nrolls, rollMin, rollMax)

    def stimulusCorrelation(self, stimulus, nrolls=0, rollMin=0, rollMax=None):
        """
        Computes the correlation between imaging responses and a given stimulus for each unit (row in data matrix)
        Args:
            stimulus: The stimulus regressor. Either 1D with the same length as rows in the imaging data matrix, or same
             size as imaging data matrix.
            nrolls: Optionally also computes mean and standard deviation of shuffle correlations
            rollMin: The minimum shuffle distance
            rollMax: The maximum shuffle distance. If None will be set to the number of columns, i.e. timepoints

        Returns: For each unite the correlation of its activity to the motor output and optionally shuffled mean
                 and standard deviation

        """
        if stimulus.shape != self.RawData.shape:
            # try to expand stimulus to same size as RawData
            if stimulus.ndim == 2:
                if stimulus.shape[0] != 1 or stimulus.shape[1] != self.NFrames:
                    raise ValueError("Stimulus shape does not match (1,NFrames)")
                stimulus = np.repeat(stimulus, self.NUnits, 0)
            elif stimulus.ndim == 1:
                if stimulus.size != self.NFrames:
                    raise ValueError("Need one stimulus frame per imaging frame")
                stimulus = np.repeat(stimulus[None, :], self.NUnits, 0)
            else:
                raise ValueError("Incompatible stimulus dimensionality")
        # now the stimulus should have the same shape as the data
        return self.correlate(self.RawData, stimulus, nrolls, rollMin, rollMax)

    def repeatAveragedTimeseries(self, n_repeats, n_hangover=1):
        """
        Computes a repeat average across the raw timeseries
        Args:
            n_repeats: The number of experimental repeats
            n_hangover: The number of "extra frames" tucked on the end

        Returns:

        """
        return self.computeRepeatAverage(self.RawData, n_repeats, n_hangover)

    def fourierEntropy(self, nshuffles):
        """
        Computes the entropy of the fourier spectrum magnitudes and optionally and associated level of significance
        Args:
            nshuffles: The number of shuffles to perform

        Returns:
            [0]: For each unit the entropy of fourier magnitudes
            [1]: The level of significance (p-value) in comparison to shuffled distribution

        """

        ent_real = self.fourierEnt(self.RawData)
        if nshuffles < 2:
            return ent_real, None
        ent_shuff = np.zeros((self.RawData.shape[0], nshuffles), dtype=np.float32)
        for i in range(nshuffles):
            # we want to randomize elements but *only* along the time axis
            shuffle = np.random.choice(np.arange(self.RawData.shape[1]), self.RawData.shape[1])
            sh = self.RawData[:, shuffle]
            ent_shuff[:, i] = self.fourierEnt(sh)
        pv = np.sum(ent_real[:, None] >= ent_shuff, 1) / nshuffles
        return ent_real, pv

    @staticmethod
    def fourierEnt(trace):
        magnitudes = np.absolute(np.fft.rfft(ImagingData.zscore(trace), axis=1))
        sm = np.sum(magnitudes, 1, keepdims=True)
        magnitudes /= sm
        e = magnitudes * np.log2(magnitudes)
        return -1 * np.nansum(e, 1)

    @staticmethod
    def meanSubtract(m):
        """
        Returns a matrix with the row-means subtracted out
        """
        return m - np.mean(m, 1, keepdims=True)

    @staticmethod
    def zscore(m):
        """
        Returns a matrix with each row being zscored
        """
        return (m - np.mean(m, 1, keepdims=True)) / np.std(m, 1, keepdims=True)

    @staticmethod
    def correlate(m1, m2, nrolls, rollMin, rollMax):
        """
            Efficiently computes the row-wise correlation and optionally shuffles for m1 and m2
            Args:
                m1: First matrix
                m2: Second matrix
                nrolls: Optionally also computes mean and standard deviation of shuffle correlations
                rollMin: The minimum shuffle distance
                rollMax: The maximum shuffle distance. If None will be set to the number of columns, i.e. timepoints

            Returns: For each row in m1 and m2 their pairwise correlation

            """
        if m1.shape != m2.shape:
            raise ValueError("Both matrices need to have same shape")
        ms1 = ImagingData.meanSubtract(m1)
        ms2 = ImagingData.meanSubtract(m2)
        dot = np.sum(ms1 * ms2, 1, keepdims=True)
        n1 = np.linalg.norm(ms1, axis=1, keepdims=True)
        n2 = np.linalg.norm(ms2, axis=1, keepdims=True)
        nprod = n1 * n2
        corr_real = dot / nprod
        if nrolls < 2:
            return corr_real, None, None
        if rollMax is None:
            rollMax = ms1.shape[1]
        if rollMax <= rollMin:
            raise ValueError("rollMax has to be strictly greater than rollMin")
        c_shuff = np.zeros((m1.shape[0], nrolls), dtype=np.float32)
        rl = np.random.randint(rollMin, rollMax, nrolls)
        for i in range(nrolls):
            sh = np.roll(ms1, rl[i], 1)
            # NOTE: The roll along rows does not affect the row-mean and does not affect the norm
            dot = np.sum(sh * ms2, 1, keepdims=True)
            c_shuff[:, i] = (dot / nprod).flatten()  # unfortunately numpy cannot assign shape(n,1) to a single column
        return corr_real.flatten(), np.mean(c_shuff, 1), np.std(c_shuff, 1)

    @staticmethod
    def computeRepeatAverage(timeseries, n_repeats, n_hangover):
        l = timeseries.shape[1]
        if (l-n_hangover) % n_repeats != 0:
            raise ValueError("Can't divide timeseries into the given number of repeats")
        repLen = (l-n_hangover) // n_repeats
        byBlock = np.reshape(timeseries[:, :l - n_hangover], (timeseries.shape[0], repLen, n_repeats), order='F')
        return np.mean(byBlock, 2)

    @staticmethod
    def computeTraceFourierTransform(timeseries, aaFilterWindow, frameRate, start=0, end=None):
        """
        Computes the fourier transform along the rows of a given matrix, optionally only within a given window
        Args:
            timeseries: Matrix of imaging timeseries with n-units rows and n-timepoints columns
            aaFilterWindow: Size of the anti-aliasing filter applied to the timeseries
            frameRate: The imaging framerate in Hz to obtain frequencies corresponding to transform
            start: Start-frame (inclusive).
            end: End-frame (exclusive). If None set to end of trace

        Returns:
            [0]: rfft of the timeseries along rows
            [1]: The associated frequencies

        """
        if aaFilterWindow > 0:
            fts = gaussian_filter(timeseries, (0, aaFilterWindow))
        else:
            fts = timeseries
        if end is None:
            end = timeseries.shape[1]
        rft = np.fft.rfft(fts[:, start:end], axis=1)
        freqs = np.linspace(0, frameRate/2, rft.shape[1])
        return rft, freqs


class RepeatExperiment(ImagingData):
    """
    Represents any experiment with a per-plane repeated structure and a defined
    set of stimulus and motor regressors
    """
    def __init__(self, imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs):
        """
        Creates a new RepeatExperiment
        Args:
            imagingData: Imaging data across all units (rows)
            swimVigor: Swim vigor matrix
            preFrames: Number of pre-stimulus frames
            stimFrames: Number of stimulus frames
            postFrames: Number of post-stimulus frames
            nRepeats: Number of per-plane repeats
            caTimeConstant: The time-constant of the used calcium indicator
        """
        super().__init__(imagingData, swimVigor)
        self.preFrames = preFrames
        self.stimFrames = stimFrames
        self.postFrames = postFrames
        self.nRepeats = nRepeats
        self.caTimeConstant = caTimeConstant
        # the following parameters would only very rarely change
        if "nHangoverFrames" in kwargs:
            self.nHangoverFrames = kwargs["nHangoverFrames"]
        else:
            self.nHangoverFrames = 1
        self.sine_amp = 0.51  # the amplitude of the sine-wave relativ to its offset
        if "frameRate" in kwargs:
            self.frameRate = kwargs["frameRate"]
        else:
            self.frameRate = 2.4  # the imaging framerate
        self.stimOn = np.array([])
        self.stimOff = np.array([])

    def computeStimulusRegressors(self):
        pass

    def onStimulusCorrelation(self, nrolls):
        """
        For each unit computes correlation with our on stimulus regressor and optionally shuffle measures
        Args:
            nrolls: Number of rolls to perform when computing shuffles

        Returns:
            [0]: Correlations with the ON regressor
            [1]: Average shuffle correlation
            [2]: Standard deviation of shuffle correlations

        """
        return self.stimulusCorrelation(self.stimOn, nrolls, self.preFrames//3, self.RawData.shape[1]-self.preFrames//3)

    def offStimulusCorrelation(self, nrolls):
        """
        For each unit computes correlation with our off stimulus regressor and optionally shuffle measures
        Args:
            nrolls: Number of rolls to perform when computing shuffles

        Returns:
            [0]: Correlations with the OFF regressor
            [1]: Average shuffle correlation
            [2]: Standard deviation of shuffle correlations

        """
        return self.stimulusCorrelation(self.stimOff, nrolls, self.preFrames//3,
                                        self.RawData.shape[1]-self.preFrames//3)

    def motorCorrelation(self, nrolls):
        """
            For each unit computes correlation with the swim vigor
            Args:
                nrolls: Number of rolls to perform when computing shuffles

            Returns:
                [0]: Correlations with the swim vigor regressor
                [1]: Average shuffle correlation
                [2]: Standard deviation of shuffle correlations

            """
        return super().motorCorrelation(nrolls, self.preFrames//3, self.RawData.shape[1]-self.preFrames//3)

    def stimEffect(self, trace):
        """
        Computes the repeat-average difference of average activity between stimulus and pre-stimulus periods as a
        fraction of pre-stimulus standard deviations
        """
        l = trace.shape[1]
        if (l - self.nHangoverFrames) % self.nRepeats != 0:
            raise ValueError("Can't divide timeseries into the given number of repeats")
        repLen = (l - self.nHangoverFrames) // self.nRepeats
        byBlock = np.reshape(trace[:, :l - self.nHangoverFrames], (trace.shape[0], repLen, self.nRepeats), order='F')
        pre = byBlock[:, :self.preFrames, :]
        m_pre = np.mean(pre, 1)
        stim = byBlock[:, self.preFrames:, :]
        m_stim = np.mean(stim, 1)
        d_pre = np.std(pre, 1)
        d_ps = np.abs(m_pre - m_stim)
        return np.mean(d_ps / d_pre, 1)

    def computeStimulusEffect(self, nrolls):
        se_real = self.stimEffect(self.RawData)
        if nrolls < 2:
            return se_real, None, None
        rolls = np.random.randint(self.preFrames//3, self.RawData.shape[1]-self.preFrames//3, nrolls)
        se_shuff = np.zeros((self.RawData.shape[0], nrolls), dtype=np.float32)
        for i in range(nrolls):
            r = np.roll(self.RawData, rolls[i], 1)
            se_shuff[:, i] = self.stimEffect(r)
        return se_real, np.mean(se_shuff, 1), np.std(se_shuff, 1)

    def computeVarianceQualScore(self):
        l = self.RawData.shape[1]
        if (l - self.nHangoverFrames) % self.nRepeats != 0:
            raise ValueError("Can't divide timeseries into the given number of repeats")
        repLen = (l - self.nHangoverFrames) // self.nRepeats
        byBlock = np.reshape(self.RawData[:, :l - self.nHangoverFrames],
                             (self.RawData.shape[0], repLen, self.nRepeats), order='F')
        var_blockAvg = np.var(np.mean(byBlock, 2), 1)
        avg_blockVar = np.mean(np.var(byBlock, 1), 1)
        return var_blockAvg / avg_blockVar


class SOORepeatExperiment(RepeatExperiment):
    """
    Represents a sine-on-off stimulation experiment with per-plane repeats
    """
    def __init__(self, imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs):
        """
        Creates a new sine-on-off repeat experiment class
        Args:
            imagingData: Imaging data across all units (rows)
            swimVigor: Swim vigor matrix
            preFrames: Number of pre-stimulus frames
            stimFrames: Number of stimulus frames
            postFrames: Number of post-stimulus frames
            nRepeats: Number of per-plane repeats
            caTimeConstant: The time-constant of the used calcium indicator
        """
        super().__init__(imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs)
        self.stimFrequency = 0.1  # the sine stimulus frequency in Hz
        self.transOn = np.array([])
        self.transOff = np.array([])
        self.stimFFTGap = int(self.frameRate * 20)
        # construct our stimulus regressors
        self.computeStimulusRegressors()

    def computeStimulusRegressors(self):
        """
        Computes all relevant stimulus reqgressors and assigns them to the class can be called after
        class parameters were changed
        """
        post_start = self.preFrames + self.stimFrames
        step_len = int(15 * self.frameRate)
        rep_len = (self.RawData.shape[1] - self.nHangoverFrames) // self.nRepeats
        stimOn = np.zeros(rep_len, dtype=np.float32)
        sine_frames = np.arange(post_start - self.preFrames)
        sine_time = sine_frames / self.frameRate
        stimOn[self.preFrames:post_start] = 1 + self.sine_amp * np.sin(sine_time * 2 * np.pi * self.stimFrequency)
        stimOn[post_start + step_len:post_start + 2 * step_len] = 1
        stimOn[post_start + 3 * step_len:post_start + 4 * step_len] = 1
        # expand by number of repetitions and add hangover frame(s)
        stimOn = np.tile(stimOn, self.nRepeats)
        stimOn = np.append(stimOn, np.zeros((self.nHangoverFrames, 1), dtype=np.float32))
        # NOTE: heating half-time inferred from phase shift observed in "responses" in pure RFP stack
        # half-time inferred to be: 891 ms
        # => time-constant beta = 0.778
        # NOTE: If correct, this means that heating kinetics in this set-up are about half way btw. freely
        # swimming kinetics and kinetics in the embedded setup. Maybe because of a) much more focuses beam
        # and b) additional heat-sinking by microscope objective
        # alpha obviously not determined
        # to simplify import, use same convolution method as for calcium kernel instead of temperature
        # prediction.
        stimOn = CaConvolve(stimOn, 0.891, self.frameRate)
        # derive transient regressors, then convolve
        transOn = np.r_[0, np.diff(stimOn)]
        transOn[transOn < 0] = 0
        transOff = np.r_[0, np.diff(-1*stimOn)]
        transOff[transOff < 0] = 0
        stimOn = CaConvolve(stimOn, self.caTimeConstant, self.frameRate)
        stimOn = (stimOn / stimOn.max()).astype(np.float32)
        transOn = CaConvolve(transOn, self.caTimeConstant, self.frameRate)
        transOn = (transOn / transOn.max()).astype(np.float32)
        transOff = CaConvolve(transOff, self.caTimeConstant, self.frameRate)
        transOff = (transOff / transOff.max()).astype(np.float32)
        stimOff = 1 - stimOn
        self.stimOn = stimOn
        self.stimOff = stimOff
        self.transOn = transOn
        self.transOff = transOff

    def computeFourierMetrics(self, aggregate=True):
        """
        Computes the fourier transform and various derived metrics for each unit
        Args:
            aggregate: Whether to perform 2-fold aggregation of the stimulus trace in addtion to repeat averaging

        Returns:
            [0]: The fourier transform of the stimulus period
            [1]: The associated frequencies
            [2]: The magnitude at the stimulus frequency
            [3]: The magnitude fraction at the stimulus frequency
            [4]: The phase angle at the stimulus frequency

        """
        avg = self.repeatAveragedTimeseries(self.nRepeats, self.nHangoverFrames)
        sl = avg[:, self.preFrames+self.stimFFTGap:self.preFrames+self.stimFrames]
        if aggregate:
            pl = round(1 / self.stimFrequency * self.frameRate)
            if (sl.shape[1] / pl) % 2 == 0:
                sl = self.aggregate(sl)
            else:
                warnings.warn('Could not aggregate for fourier due to phase alignment mismatch')
        rfft, freqs = self.computeTraceFourierTransform(self.meanSubtract(sl), self.frameRate/8, self.frameRate)
        ix = np.argmin(np.absolute(self.stimFrequency - freqs))  # index of bin which contains our stimulus frequency
        mag_atStim = np.absolute(rfft[:, ix])
        mfrac_atStim = mag_atStim / np.sum(np.absolute(rfft), 1)
        ang_atStim = np.angle(rfft[:, ix])
        return rfft, freqs, mag_atStim, mfrac_atStim, ang_atStim

    def computeStimulusLocking(self):
        """
        For each unit computes the magnitude fraction at the stimulus frequency during stimulation over the magnitude
        at the stimulus frequency during the pre-stimulation period. Data is repeat-averaged but not aggregated
        """
        avg = self.repeatAveragedTimeseries(self.nRepeats, self.nHangoverFrames)
        stim = avg[:, self.preFrames+self.stimFFTGap:self.preFrames+self.stimFrames]
        pre = avg[:, self.stimFFTGap:self.preFrames]
        rfft_stim, freqs_stim = self.computeTraceFourierTransform(self.meanSubtract(stim), self.frameRate/8,
                                                                  self.frameRate)
        ix_stim = np.argmin(np.absolute(self.stimFrequency - freqs_stim))
        mag_atStim_stim = np.absolute(rfft_stim[:, ix_stim]) / rfft_stim.shape[1]
        # mfrac_stimPeriod = mag_atStim_stim / np.sum(np.absolute(rfft_stim), 1)
        rfft_pre, freqs_pre = self.computeTraceFourierTransform(self.meanSubtract(pre), self.frameRate / 8,
                                                                self.frameRate)
        ix_pre = np.argmin(np.absolute(self.stimFrequency - freqs_pre))
        mag_atStim_pre = np.absolute(rfft_pre[:, ix_pre]) / rfft_pre.shape[1]
        # mfrac_prePeriod = mag_atStim_pre / np.sum(np.absolute(rfft_pre), 1)
        return np.nan_to_num(mag_atStim_stim - mag_atStim_pre)

    def computeFourierFractionShuffles(self, nrolls, aggregate=True):
        """
        Computes shuffle mean and standard deviation of the fourier magnitude fraction at the stimulus frequency
        Args:
            nrolls: The number of shuffles to perform
            aggregate: Whether to perform 2-fold aggregation of the stimulus trace in addtion to repeat averaging

        Returns:
            [0]: The average of the shuffles
            [1]: The standard deviation of the shuffles

        """
        rolls = np.random.randint(self.preFrames//3, self.RawData.shape[1]-self.preFrames//3, nrolls)
        mfracs = np.zeros((self.RawData.shape[0], nrolls), dtype=np.float32)
        for i in range(nrolls):
            r = np.roll(self.RawData, rolls[i], 1)
            a = self.computeRepeatAverage(r, self.nRepeats, self.nHangoverFrames)
            a = a[:, self.preFrames+self.stimFFTGap:self.preFrames+self.stimFrames]
            if aggregate:
                a = self.aggregate(a)
            rfft, freqs = self.computeTraceFourierTransform(self.meanSubtract(a), self.frameRate/8, self.frameRate)
            ix = np.argmin(np.absolute(self.stimFrequency - freqs))
            mas = np.absolute(rfft[:, ix])
            s = np.sum(np.absolute(rfft), 1)
            mfracs[:, i] = mas / s
        return np.mean(mfracs, 1), np.std(mfracs, 1)

    @staticmethod
    def aggregate(timeseries):
        return ImagingData.computeRepeatAverage(timeseries, 2, 0)


class SLHRepeatExperiment(SOORepeatExperiment):
    """
    Represents a sine-on-off stimulation experiment with per-plane repeats in which the first
    on stimuli during the post phase are at average sine stimulus strength and the second at
    maximum sine stimulus strength
    """
    def __init__(self, imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs):
        """
        Creates a new sine-on-off lo-hi repeat experiment class
        Args:
            imagingData: Imaging data across all units (rows)
            swimVigor: Swim vigor matrix
            preFrames: Number of pre-stimulus frames
            stimFrames: Number of stimulus frames
            postFrames: Number of post-stimulus frames
            nRepeats: Number of per-plane repeats
            caTimeConstant: The time-constant of the used calcium indicator
        """
        super().__init__(imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs)

    def computeStimulusRegressors(self):
        """
        Computes all relevant stimulus reqgressors and assigns them to the class can be called after
        class parameters were changed
        """
        post_start = self.preFrames + self.stimFrames
        step_len = int(15 * self.frameRate)
        rep_len = (self.RawData.shape[1] - self.nHangoverFrames) // self.nRepeats
        stimOn = np.zeros(rep_len, dtype=np.float32)
        sine_frames = np.arange(post_start - self.preFrames)
        sine_time = sine_frames / self.frameRate
        stimOn[self.preFrames:post_start] = 1 + self.sine_amp * np.sin(sine_time * 2 * np.pi * self.stimFrequency)
        stimOn[post_start + step_len:post_start + 2 * step_len] = 1
        stimOn[post_start + 3 * step_len:post_start + 4 * step_len] = 1 + self.sine_amp
        # expand by number of repetitions and add hangover frame(s)
        stimOn = np.tile(stimOn, self.nRepeats)
        stimOn = np.append(stimOn, np.zeros((self.nHangoverFrames, 1), dtype=np.float32))
        # NOTE: heating half-time inferred from phase shift observed in "responses" in pure RFP stack
        # half-time inferred to be: 891 ms
        # => time-constant beta = 0.778
        # NOTE: If correct, this means that heating kinetics in this set-up are about half way btw. freely
        # swimming kinetics and kinetics in the embedded setup. Maybe because of a) much more focuses beam
        # and b) additional heat-sinking by microscope objective
        # alpha obviously not determined
        # to simplify import, use same convolution method as for calcium kernel instead of temperature
        # prediction.
        stimOn = CaConvolve(stimOn, 0.891, self.frameRate)
        # derive transient regressors, then convolve
        transOn = np.r_[0, np.diff(stimOn)]
        transOn[transOn < 0] = 0
        transOff = np.r_[0, np.diff(-1*stimOn)]
        transOff[transOff < 0] = 0
        stimOn = CaConvolve(stimOn, self.caTimeConstant, self.frameRate)
        stimOn = (stimOn / stimOn.max()).astype(np.float32)
        transOn = CaConvolve(transOn, self.caTimeConstant, self.frameRate)
        transOn = (transOn / transOn.max()).astype(np.float32)
        transOff = CaConvolve(transOff, self.caTimeConstant, self.frameRate)
        transOff = (transOff / transOff.max()).astype(np.float32)
        stimOff = 1 - stimOn
        self.stimOn = stimOn
        self.stimOff = stimOff
        self.transOn = transOn
        self.transOff = transOff


class HeatPulseExperiment(RepeatExperiment):
    """
    Represents an experiment with heat-pulse presentation during stimulus phase
    10s base current, followed by 350ms 0 current followed by 300ms peak current
    and subsequent return to baseline
    """

    def __init__(self, imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs):
        super().__init__(imagingData, swimVigor, preFrames, stimFrames, postFrames, nRepeats, caTimeConstant, **kwargs)
        if "baseCurrent" in kwargs:
            self.baseCurrent = kwargs["baseCurrent"]
        else:
            self.baseCurrent = 71  # mW at coll output for 700 mA current
        if "peakCurrent" in kwargs:
            self.peakCurrent = kwargs["peakCurrent"]
        else:
            self.peakCurrent = 280  # mW at coll output for 2000 mA current
        self.computeStimulusRegressors()

    def computeStimulusRegressors(self):
        rep_len = (self.RawData.shape[1] - self.nHangoverFrames) // self.nRepeats
        rep_time = rep_len / self.frameRate
        # first create regressor at 1000Hz then interpolate down
        stimOn = np.zeros(int(rep_time * 1000), dtype=np.float32)
        pre_time = self.preFrames / self.frameRate
        stim_time = self.stimFrames / self.frameRate
        stimOn[int(pre_time*1000) : int((pre_time+stim_time)*1000)] = self.baseCurrent
        baseTime = 1000 * 10
        peakStart = int(pre_time*1000 + baseTime)
        peakEnd = peakStart + 500
        stimOn[peakStart:peakEnd] = self.peakCurrent
        # bin back to actual time
        i_time = np.arange(rep_len+1) / self.frameRate
        digitized = np.digitize(np.arange(stimOn.size)/1000, i_time)
        stimOn = np.array([[stimOn[digitized == i].mean() for i in range(1, i_time.size)]])
        # expand by number of repetitions and add hangover frame(s)
        stimOn = np.tile(stimOn, self.nRepeats)
        stimOn = np.append(stimOn, np.zeros((self.nHangoverFrames, 1), dtype=np.float32))
        stimOn = CaConvolve(stimOn, 0.891, self.frameRate)  # convert to temp
        stimOn = CaConvolve(stimOn, self.caTimeConstant, self.frameRate)  # convert to calcium transient
        stimOn = (stimOn / stimOn.max()).astype(np.float32)  # normalize
        stimOff = 1 - stimOn
        self.stimOn = stimOn
        self.stimOff = stimOff


class PixelGraph:

    def __init__(self,id):
        self.V = []#list of vertices
        self.ID = id#id of this graph for easy component reference

    @property
    def NPixels(self):
        return len(self.V)
# class PixelGraph


# compiled acceleration


@jit(numba.float64[:](numba.float64[:],numba.int32))
def Vigor(cumAngle,winlen=10):
    """
    Computes the swim vigor based on a cumulative angle trace
    as the windowed standard deviation of the cumAngles
    """
    s = cumAngle.size
    vig = zeros(s)
    for i in range(winlen,s):
        vig[i] = std(cumAngle[i-winlen+1:i+1])
    return vig


class TailData:

    def __init__(self, fileData, ca_timeconstant, frameRate):
        """
        Creates a new TailData object
            fileData: Matrix loaded from tailfile
        """
        self.scanning = fileData[:, 0] == 1
        self.scanFrame = fileData[:, 1].astype(int)
        self.scanFrame[np.logical_not(self.scanning)] = -1
        # after the last frame is scanned, the scanImageIndex will be incremented further
        # and the isScanning indicator will not immediately switch off. Therefore, if
        # the highest index frame has less than 75% of the average per-index frame-number
        # set it to -1 as well
        c = Counter(self.scanFrame[self.scanFrame != -1])
        avgCount = np.median(list(c.values()))
        maxFrame = np.max(self.scanFrame)
        if np.sum(self.scanFrame == maxFrame) < 0.75*avgCount:
            self.scanFrame[self.scanFrame == maxFrame] = -1
        self.cumAngles = np.rad2deg(fileData[:, 2])
        self.ca_original = self.cumAngles.copy()
        self.RemoveTrackErrors()
        self.vigor = Vigor(self.cumAngles, 6)  # just below the number of our minimum bout frames (8)
        # compute vigor bout threshold
        t = np.mean(self.vigor[self.vigor < 25]) + 2*np.std(self.vigor[self.vigor < 25])
        print("Vigor threshold = ", t)
        self.bouts = mb.DetectTailBouts(self.cumAngles, threshold=t, frameRate=frameRate, vigor=self.vigor)
        if self.bouts is not None and self.bouts.size == 0:
            self.bouts = None
        if self.bouts is not None:
            bs = self.bouts[:, 0].astype(int)
            self.boutFrames = self.scanFrame[bs]
        else:
            self.boutFrames = []
        self.ca_kernel = TailData.CaKernel(ca_timeconstant, frameRate)
        self.ca_timeconstant = ca_timeconstant
        self.frameRate = frameRate
        # compute tail velocities based on 10-window filtered cumulative angle trace
        fca = lfilter(np.ones(10)/10, 1, self.cumAngles)
        self.velocity = np.hstack((0, np.diff(fca)))
        self.velcty_noise = np.nanstd(self.velocity[self.velocity<4])
        # compute a time-trace assuming a constant frame-rate which starts at 0
        # for the likely first camera frame during the first acquisition frame
        # we infer this frame by going back avgCount frames from the first frame
        # of the second (!) scan frame (i.e. this should then be the first frame
        # of the first scan frame)
        frames = np.arange(self.cumAngles.size)
        first_frame = np.min(frames[self.scanFrame == 1]) - avgCount.astype(int)
        frames -= first_frame
        self.frameTime = (frames / frameRate).astype(np.float32)
        # create bout-start trace at original frame-rate
        self.starting = np.zeros_like(self.frameTime)
        if self.bouts is not None:
            self.starting[self.bouts[:, 0].astype(int)] = 1


    def RemoveTrackErrors(self):
        """
        If part of the agarose gel boundary is visible in the frame
        the tracker will occasionally latch onto it for single frames.
        Tries to detect these instances and corrects them
        """
        vicinity = np.array([-2, -1, 1, 2])
        for i in range(2, self.cumAngles.size-2):
            d_pre = self.cumAngles[i] - self.cumAngles[i-1]
            d_post = self.cumAngles[i+1] - self.cumAngles[i]
            # the current point is surrounded by two similar cumulative angles
            # that are both 45 degrees away in the same direction
            if (d_pre > 30 and d_post < -30) or (d_pre < -30 and d_post > 30):
                # the angles in the vicinity of the current point are similar
                if np.ptp(self.cumAngles[vicinity+i]) < 10:
                    self.cumAngles[i] = (self.cumAngles[i-1] + self.cumAngles[i+1])/2

    @property
    def PerFrameVigor(self):
        """
        For each scan frame returns the average
        swim vigor
        """
        sf = np.unique(self.scanFrame)
        sf = sf[sf != -1]
        sf = np.sort(sf)
        conv_vigor = np.convolve(self.vigor, self.ca_kernel, mode='full')[:self.vigor.size]
        pfv = np.zeros(sf.size)
        for i, s in enumerate(sf):
            pfv[i] = np.mean(conv_vigor[self.scanFrame == s])
        return pfv

    @property
    def BoutStartsEnds(self):
        if self.bouts is None:
            return None, None
        else:
            return self.bouts[:, 0].astype(int), self.bouts[:, 1].astype(int)

    @property
    def ConvolvedStarting(self):
        """
        Returns a convolved version of the camera frame-rate
        bout start trace
        """
        return np.convolve(self.starting, self.ca_kernel, mode='full')[:self.starting.size]

    @property
    def FrameBoutStarts(self):
        """
        Returns a convolved per-frame bout-start trace
            image_freq: Imaging frequency
        """
        if self.bouts is None:
            sf = np.unique(self.scanFrame)
            sf = sf[sf != -1]
            return np.zeros(sf.size)
        conv_starting = self.ConvolvedStarting
        # collect all valid scan-frames
        sf = np.unique(self.scanFrame)
        sf = sf[sf != -1]
        sf = np.sort(sf)
        per_f_s = np.zeros(sf.size)
        for i, s in enumerate(sf):
            per_f_s[i] = np.mean(conv_starting[self.scanFrame == s])
        return per_f_s

    def PlotBouts(self):
        bs, be = self.BoutStartsEnds
        with sns.axes_style('white'):
            pl.figure()
            pl.plot(self.cumAngles,label='Angle trace')
            if bs is not None:
                pl.plot(bs, self.cumAngles[bs], 'r*', label='Starts')
                pl.plot(be, self.cumAngles[be], 'k*', label='Ends')
            pl.ylabel('Cumulative tail angle')
            pl.xlabel('Frames')
            sns.despine()

    @staticmethod
    def LoadTailData(filename, ca_timeConstant, frameRate=100):
        try:
            data = np.genfromtxt(filename, delimiter='\t')
        except (IOError, OSError):
            return None
        return TailData(data, ca_timeConstant, frameRate)

    @staticmethod
    def CaKernel(tau, frameRate):
        """
        Creates a calcium decay kernel for the given frameRate
        with the given half-life in seconds
        """
        fold_length = 5  # make kernel length equal to 5 half-times (decay to 3%)
        klen = int(fold_length*tau*frameRate)
        tk = np.linspace(0, fold_length*tau, klen, endpoint=False)
        k = 2**(-1*tk/tau)
        k = k / k.sum()
        return k


# TailData


class TailDataDict:

    def __init__(self, ca_timeConstant=1.796, frameRate=100):
        self._td_dict = dict()
        self.ca_timeConstant = ca_timeConstant
        self.frameRate = frameRate

    def __getitem__(self, item):
        if type(item) != str:
            raise TypeError('Indexer needs to be string (filename)')
        if len(item) > 4 and item[-4:] == 'tail':
            tf = item
        else:
            tf = self.tailFile(item)
        if tf in self._td_dict:
            return self._td_dict[tf]
        else:
            try:
                td = TailData.LoadTailData(tf, self.ca_timeConstant, self.frameRate)
            except:
                raise KeyError('Could not find taildata for file')
            if td is None:
                raise KeyError('Could not find taildata for file')
            self._td_dict[tf] = td
            return td

    @staticmethod
    def tailFile(tif_name):
        return tif_name[:-6]+'.tail'

    @property
    def fileNames(self):
        return self._td_dict.keys()
# TailDataDict


def UiGetFile(filetypes=[('Tiff stack', '.tif;.tiff')], multiple=False):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    options = {}
    options['filetypes'] = filetypes
    options['multiple'] = multiple
    Tkinter.Tk().withdraw()  # Close the root window
    return tkFileDialog.askopenfilename(**options)
# UiGetFile


def OpenStack(filename):
    """
    Load image stack from tiff-file
    """
    im = Image.open(filename)
    stack = np.empty((im.n_frames, im.size[1], im.size[0]), dtype=np.float32)
    # loop over frames and assign
    for i in range(im.n_frames):
        im.seek(i)
        stack[i, :, :] = np.array(im)
    im.close()
    return stack


def CaConvolve(trace, ca_timeconstant, frame_rate):
    kernel = TailData.CaKernel(ca_timeconstant, frame_rate)
    return np.convolve(trace, kernel)[:trace.size]
    

def FilterStack(stack):
    """
    8-connected neighborhood filter of pixel timeseries. Pixel itself
    contributes 1/2 each connected neighbor 1/16
    """
    width = stack.shape[1]
    height = stack.shape[2]
    for y in range(1, height-1):
        for x in range(1, width-1):
            trace = np.zeros(stack.shape[0])
            for jit_x in range(-1,2):
                for jit_y in range(-1,2):
                    if jit_x == 0 and jit_y == 0:
                        trace = trace + 0.5 * stack[:,x,y]
                    else:
                        trace = trace + 1/16 * stack[:,x+jit_x,y+jit_y]
            stack[:, x, y] = trace
    return stack  # should be in place anyways

def FilterStackGaussian(stack, sigma=1):
    """
    Performs per-plane gaussian filter (assumed axis0=time)
    using the given standard deviation in pixels
    """
    out = np.zeros_like(stack, dtype='float')
    for t in range(stack.shape[0]):
        out[t, :, :] = gaussian_filter(stack[t, :, :].astype(float), sigma, multichannel=False)
    return out
    
def DeltaFOverF(stack):
    """
    Transforms stack into delta-f over F metric
    """
    F = np.mean(stack, 0)[None, :, :]
    F.repeat(stack.shape[0],0)
    return (stack-F)/F
    
def Per_PixelFourier(stack, tstart=None, tend=None):
    """
    For reach pixel performs a fourier transform of the time-series (axis 0)
    between the given start and end frames. Resulting stack will have fourier
    components in axis 0, replacing time components.
    """
    if tstart is None:
        tstart = 0
    if tend is None:
        tend = stack.shape[0]  # tend is exclusive
    if tend <= tstart:
        raise ValueError("tend has to be larger than tstart")
    flen = (tend-tstart)//2 + 1
    f_mag_stack = np.zeros((flen,stack.shape[1],stack.shape[2]))
    f_phase_stack = np.zeros_like(f_mag_stack)
    for y in range(stack.shape[2]):
        for x in range(stack.shape[1]):
            if np.sum(np.isnan(stack[:, x, y])) == 0:
                # compute transform on mean-subtracted trace - don't want to punish thresholding
                # for bright pixels
                transform = np.fft.rfft(stack[tstart:tend, x, y]-np.mean(stack[tstart:tend, x, y]))
                f_mag_stack[:, x, y] = np.absolute(transform)
                f_phase_stack[:, x, y] = np.angle(transform)
            else:
                f_mag_stack[:, x, y] = np.full(flen, np.NaN)
                f_phase_stack[:, x, y] = np.full(flen, np.NaN)
    return f_mag_stack, f_phase_stack


def ZScore_Stack(stack):
    """
    Replaces each pixel-timeseries/frequency-series with its
    corresponding z-score
    """
    avg = np.mean(stack, 0)[None, :, :]
    std = np.std(stack, 0)[None, :, :]
    # do not introduce NaN's in zero rows (which will have avg=std=0) but rather keep
    # them as all zero
    std[avg == 0] = 1
    return (stack-avg.repeat(stack.shape[0], 0))/std.repeat(stack.shape[0], 0)

def Threshold_Zsc(stack, nstd, plane):
    """
    Takes the ZScore of stack along axis 0 and thresholds the indicated plane
    such that all pixels above nstd will be equal to the zscore all others 0.
    The thresholded single plane is returned
    """
    # TODO: Since discrete FFT will smear out signals should probably allow passing multiple planes
    zsc = ZScore_Stack(stack)
    im_th = zsc[plane, :, :]
    im_th[im_th < nstd] = 0
    return im_th


def ConnectedComponents(image):
    """
    Using breadth-first search builds connected component graphs
    of all non-0 pixel regions
        image: The image to search
        RETURNS:
            [0]: A list of PixelGraphs, one entry for reach connected component
            [1]: A int32 image of same dimensions of image with each pixel
                labeled according to the id of the component graph it belongs to
    """
    def BFS(image, visited, sourceX, sourceY, color):
        """
        Performs breadth first search on image
        given (sourceX,sourceY) as starting pixel
        coloring all visited pixels in color
        """
        pg = PixelGraph(color)
        Q = deque()
        Q.append((sourceX, sourceY))
        visited[sourceX, sourceY] = color  # mark source as visited
        while len(Q) > 0:
            v = Q.popleft()
            x = v[0]
            y = v[1]
            pg.V.append(v)  # add current vertex to pixel graph
            # add non-visited neighborhood to queue
            for xn in range(x-1, x+2):  # x+1 inclusive!
                for yn in range(y-1, y+2):
                    if xn < 0 or yn < 0 or xn >= image.shape[0] or yn >= image.shape[1]:  # outside image dimensions
                        continue
                    if (not visited[xn, yn]) and image[xn, yn] > 0:
                        Q.append((xn, yn))  # add non-visited above threshold neighbor
                        visited[xn, yn] = color  # mark as visited
        return pg

    visited = np.zeros_like(image, dtype=int)  # indicates visited pixels > 0
    conn_comps = []  # list of pixel graphs
    # loop over pixels and initiate bfs whenever we encouter
    # a pixel that is non-zero and which has not yet been visited
    curr_color = 1  # id counter of connected components
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (not visited[x, y]) and image[x, y]>0:
                conn_comps.append(BFS(image, visited, x, y, curr_color))
                curr_color += 1
    return conn_comps, visited


def AvgNeighbhorCorrelations(stack, dist=2, predicate=None):
    """
    Returns a 2D image which for each pixel in stack
    has the average correlation of that pixel's time-series
    with the timeseries of it's neighbors up to dist pixels
    away from the current pixel.
    Predicate is an optional function that takes an x/y coordinate
    pair as argument and returns whether to compute the correlation.
    """
    if dist < 1:
        raise ValueError('Dist has to be at least 1')
    im_corr = np.zeros((stack.shape[1], stack.shape[2]), dtype=np.float32)
    corr_buff = dict()  # buffers computed correlations to avoid computing the same pairs multiple times!
    for x in range(stack.shape[1]):
        for y in range(stack.shape[2]):
            if (predicate is not None) and (not predicate(x, y)):  # pixel is excluded
                continue
            c_sum = []
            for dx in range(-1*dist, dist+1):  # dx=dist inclusive!
                for dy in range(-1*dist, dist+1):
                    if dx == 0 and dy == 0:  # original pixel
                        continue
                    if x+dx < 0 or y+dy < 0 or x+dx >= im_corr.shape[0] or y+dy >= im_corr.shape[1]:  # outside of image
                        continue
                    p_src = (x, y)
                    p_des = (x+dx, y+dy)
                    if (p_src, p_des) in corr_buff:
                        c_sum.append(corr_buff[(p_src, p_des)])
                    else:
                        cval = np.corrcoef(stack[:, x, y], stack[:, x+dx, y+dy])[0, 1]
                        corr_buff[(p_des, p_src)] = cval
                        c_sum.append(cval)
            if len(c_sum) > 0 and not np.all(np.isnan(c_sum)):
                im_corr[x, y] = np.nanmean(c_sum)
    im_corr[np.isnan(im_corr)] = 0
    return im_corr

# from scipy.ndimage.measurements import center_of_mass


def ComputeAlignmentShift(stack, index, sum_stack):
    """
    For the slice in stack identified by index computes
    the x (row) and y (column) shift that corresponds to
    the best alignment of stack[index,:,:] to the re-
    mainder of the stack
    """
    # shift_x = np.zeros(stack.shape[0])#best x-shift of each image
    # shift_y = np.zeros_like(shift_x)#best y-shift of each image
    # max_corr = np.zeros_like(shift_x)#un-normalized correlation at best shift

    def ms(slice):
        return slice-np.mean(slice)

    # remove current slice from the sum
    sum_stack -= stack[index, :, :]

    exp_x, exp_y = stack.shape[1]//2, stack.shape[2]//2  # these are the indices in the cross-correlation matrix that correspond to 0 shift
    c = cv2.filter2D(ms(sum_stack), -1, ms(stack[index, :, :]))
    # NOTE: Center of mass instead of maximum may be better IFF the eye has been
    # masked out of the stack. But it seems to lead to quite large distortions
    # otherwise. Namely, quality scores get substantially worse in slices with
    # the eye present after center-of-mass alignment than after peak alignment.
    # However, after masking out the eye, center-of-mass does seem to produce
    # slightly better alignments
    # c[c<0] = 0 #this line is necessary when using center-of-mass for shift!
    x, y = np.unravel_index(np.argmax(c), c.shape)  # center_of_mass(c)#
    shift_x = int(x-exp_x)
    shift_y = int(y-exp_y)
    # return sum_stack so that outer scope can add shifted verion back in
    return shift_x, shift_y, sum_stack


def ReAlign(stack, maxShift):
    """
    Re-positions every slice in stack by the following iterative
    procedure:
    Compute sum over stack excluding current slice
    Compute best-aligned x-shift and y-shift of current slice to sum-stack
    Shift the slice by x-shift and y-shift within the stack
    Increment current slice and repeat from top
    In addition to decrease the influence of imaging noise on shifts
    allows temporal filtering of the stack by filterT slices
    """
    def Shift2Index(shift, size):
        """
        Translates a given shift into the appropriate
        source and target indices
        """
        if shift < 0:
            # coordinate n in source should be n-shift in target
            source = (-1*shift, size)
            target = (0, shift)
        elif shift > 0:
            # coordinate n in source should be n+1 in target
            source = (0, -1*shift)
            target = (shift, size)
        else:
            source = (0, size)
            target = (0, size)
        return source, target

    maxShift = int(maxShift)
    x_shifts = np.zeros(stack.shape[0])
    y_shifts = np.zeros_like(x_shifts)
    re_aligned = stack.copy()
    # compute initial sum
    sum_stack = np.sum(re_aligned, 0)
    for t in range(re_aligned.shape[0]):
        xshift, yshift, sum_stack = ComputeAlignmentShift(re_aligned, t, sum_stack)
        x_shifts[t] = xshift
        y_shifts[t] = yshift
        if xshift == 0 and yshift == 0:
            continue
        if np.abs(xshift) > maxShift or np.abs(yshift) > maxShift:
            print("Warning. Slice ", t, " requires shift greater ", maxShift, " pixels. Maximally shifted")
            if xshift > maxShift:
                xshift = maxShift
            elif xshift < -1*maxShift:
                xshift = -1*maxShift
            if yshift > maxShift:
                yshift = maxShift
            elif yshift < -1*maxShift:
                yshift = -1*maxShift
        xs, xt = Shift2Index(xshift, re_aligned.shape[1])
        ys, yt = Shift2Index(yshift, re_aligned.shape[2])
        newImage = np.zeros((re_aligned.shape[1], re_aligned.shape[2]), dtype=np.float32)
        newImage[xt[0]:xt[1], yt[0]:yt[1]] = re_aligned[t, xs[0]:xs[1], ys[0]:ys[1]]
        re_aligned[t, :, :] = newImage
        # add re-aligned image to sumStack
        sum_stack += newImage
    return re_aligned, x_shifts, y_shifts


def CorrelationControl(stack, nFrames):
    """
    Sub-divides stack into nFrames blocks and cross-correlates
    each summed block to first reporting the 0-shift correlation to id
    potential movement artefacts. All slices will be z-scored to prevent
    differences in (raw) correlation values based on intensity (bleaching etc)
    """
    def zsclice(slice):
        return (slice-np.mean(slice))/np.std(slice)

    nSlices, h, w = stack.shape
    if nSlices//nFrames < 2:
        raise ValueError("Need to identify at least two nFrames sized sub-stacks in the stack")
    ix0_x, ix0_y = h//2, w//2  # coordinates of 0-shift correlation
    sum_slices = np.zeros((nSlices//nFrames, h, w), dtype=np.float32)
    correlations = np.zeros(nSlices//nFrames-1)
    for i in range(nSlices//nFrames):
        sum_slices[i, :, :] = np.sum(stack[nFrames*i:nFrames*(i+1), :, :], 0)
        if i == 0:
            z_sum = zsclice(sum_slices[0, :, :])
        elif i > 0:
            correlations[i-1] = cv2.filter2D(z_sum, -1, zsclice(sum_slices[i, :, :]))[ix0_x, ix0_y]
    return correlations, sum_slices


def ShuffleStackSpatioTemporal(stack):
    """
    Returns a version of stack that has been randomly shuffled
    along it's spatial dimensions (axis 1 and 2) as well as
    circularly permuted along it's temporal dimension (axis 0)
    """
    shuff = stack.copy()
    s0, s1, s2 = stack.shape
    for x in range(s1):
        for y in range(s2):
            xs = np.random.randint(s1)
            ys = np.random.randint(s2)
            temp = shuff[:, xs, ys].copy()  # this copy is important since otherwise we are dealing with a view not actual values!!!
            shuff[:, xs, ys] = shuff[:, x, y]
            shuff[:, x, y] = np.roll(temp, np.random.randint(s0))
    return shuff


def ShuffleStackTemporal(stack):
    """
    Returns a version of the stack that has been
    permuted at random along the time axis (axis 0)
    """
    shuff = np.empty_like(stack)
    s0, s1, s2 = stack.shape
    for x in range(s1):
        for y in range(s2):
            shuff[:, x, y] = np.random.choice(stack[:, x, y], size=s0, replace=False)
    return shuff


def cutOutliers(image, perc_cut=99.9):
    """
    Converts an image to 8-bit representation setting all pixels above
    the given percentile to 255 in order to remove extreme outliers
    """
    image = image.copy() / np.percentile(image, perc_cut)
    image[image > 1] = 1
    return image


def hessian(x):
    """
    Calculate hessian matrices with finite differences
    Parameters:
         - x : ndarray
    Returns: an array of shape (x.dim, x.ndim) + x.shape
        where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hess = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hess[k, l, :, :] = grad_kl
    return hess


def hessian_eigval_images(h):
    """
    For the hessian matrices of an image returns images of the two
    eigenvalues of each pixel's hessian matrix
    Args:
        h: Hessian matrices for each pixel of an image. Of shape [xdim, ydim, 2, 2]

    Returns:
        Two images of dimensions [xdim, ydim] corresponding to the first two eigenvalues
        of the hessian at each pixel
    """
    assert h.ndim == 4
    assert h.shape[0] == 2
    assert h.shape[1] == 2
    s_x = h.shape[2]
    s_y = h.shape[3]
    evi1 = np.zeros((s_x, s_y))
    evi2 = np.zeros_like(evi1)
    for x in range(s_x):
        for y in range(s_y):
            h_pix = h[:, :, x, y]  # hessian matrix of the given pixel
            e1, e2 = np.linalg.eig(h_pix)[0]
            evi1[x, y] = e1
            evi2[x, y] = e2
    return evi1, evi2


def PostMatchSlices(preStack, expStack, nRegions=25, radius=5, interpolate=False, regions=None):
    """
    Function to match each time-slice in expStack to a corresponding
    z-section in preStack
    Args:
        preStack: The stable z pre-stack, assumed to have 21 slices per z-plane
        expStack: The experimental time-stack
        nRegions: The number of regions to be used for matching - ignored if regions!=None
        radius: The radius of each region (region itself is square exscribing a circle of this radius) ignored if
            regions!=None
        interpolate: If true, slice positions will be interpolated 5 fold
        regions: The regions to use for identification. If None regions will be chosen randomly

    Returns:
        [0]: An array of indices that indicate the most likely position in preStack of each expStack slice
        [1]: The matrix of per-slice region correlations
        [2]: The regions used
    """

    def IdentifyRegions():
        """
        Uses simple heuristics to identify good regions to be used for matching
        Returns:
            A list of regions, with each region being a list of pixel tuples (x,y)
        """
        area = nRegions * np.pi * (radius ** 2)
        if area > sum_stack.shape[1] * sum_stack.shape[2] / 4:
            raise ValueError("Region identification fails if total region area larger than quarter of slice area")
        # sequentially pick non-overlapping region centers - in first pass three times as many as nRegions
        centers = []
        counter = 0
        while len(centers) < nRegions * 20:
            counter += 1
            if counter > nRegions * 1000:
                break
            x = np.random.randint(0, sum_stack.shape[1])
            y = np.random.randint(0, sum_stack.shape[2])
            if x < radius or y < radius or x+radius >= sum_stack.shape[1] or y+radius >= sum_stack.shape[2]:
                continue
            if np.mean(sum_stack[:, x, y]) < 2:
                continue
            if len(centers) == 0:
                centers.append((x, y))
            else:
                all_c = np.vstack(centers)
                cur_c = np.array([x, y])[None, :]
                d = np.sqrt(np.sum((all_c-cur_c)**2, 1))
                assert d.size == len(centers)
                if d.min() > 4*radius:
                    centers.append((x, y))

                # d = [np.sqrt((c[0]-x)**2 + (c[1]-y)**2) for c in centers]
                # # test if the minimal distance of all centers to the new point is larger than the radius
                # if min(d) > 4*radius:
                #     centers.append((x, y))
        regions = []
        for c in centers:
            r = []
            xs = c[0] - radius
            ys = c[1] - radius
            for x in range(xs, c[0]+radius+1):
                for y in range(ys, c[1]+radius+1):
                    r.append((x, y))
            regions.append(r)
        if len(regions) == 0:
            print("DID NOT IDENTIFY A SINGLE REGION")
            return []
        rs = GetRegionSeries(regions, sum_stack)
        # we want to keep regions that contain signal (i.e maximum value > radius**2) and whose standard deviation
        # is the largest
        regions = [r for i, r in enumerate(regions) if np.max(rs[i, :]) > radius**2]
        rs = rs[np.max(rs, 1) > radius**2, :]
        assert len(regions) == rs.shape[0]
        augmented = [(r, np.std(lfilter(np.ones(5)/5, 1, rs[i, :]))/np.mean(rs[i, :])) for i, r in enumerate(regions)]
        augmented = sorted(augmented, key=lambda x: x[1], reverse=True)
        if len(augmented) < nRegions:
            print(len(regions), " regions pass signal threshold")
            print("Only ", len(augmented), " regions could be returned")
            return [r[0] for r in augmented]
        else:
            return [augmented[i][0] for i in range(nRegions)]

    def GetRegionSeries(regions, stack):
        """
        Given a list of regions and a stack returns the time/space series for each region as a matrix
        """
        return np.vstack([np.sum(np.vstack([stack[:, v[0], v[1]] for v in r]), 0) for r in regions])

    # make pre-stack projection
    psvalid = preStack[1:, :, :]
    assert psvalid.shape[0] % 21 == 0
    sum_stack = np.zeros((preStack.shape[0] // 21, preStack.shape[1], preStack.shape[2]), dtype=np.float32)
    for i in range(sum_stack.shape[0]):
        sum_stack[i, :, :] = np.sum(preStack[i * 21:(i + 1) * 21, :, :], 0)
    if regions is None:
        regions = IdentifyRegions()
    if len(regions) == 0:
        return None, None, None
    pre_series = GetRegionSeries(regions, sum_stack)
    exp_series = GetRegionSeries(regions, expStack)
    sl_indices = np.arange(pre_series.shape[1])
    if interpolate:
        ind_interp = np.linspace(0, sl_indices.max(), sl_indices.size*5, endpoint=True)
        f_interp = interp1d(sl_indices, pre_series, axis=1)
        pre_series = f_interp(ind_interp)
        sl_indices = ind_interp
        assert pre_series.shape[0] == exp_series.shape[0]
        assert pre_series.shape[1] == sum_stack.shape[0]*5
    # time-filter experimental series
    exp_series = lfilter(np.ones(5)/5, 1, exp_series, axis=1)
    # mean-subtract and normalize columns
    pre_series -= np.mean(pre_series, 0, keepdims=True)
    pre_series /= np.linalg.norm(pre_series, axis=0, keepdims=True)
    exp_series -= np.mean(exp_series, 0, keepdims=True)
    exp_series /= np.linalg.norm(exp_series, axis=0, keepdims=True)
    slices = np.zeros(expStack.shape[0])
    corrs = np.zeros((pre_series.shape[1], expStack.shape[0]), dtype=float)
    for i in range(expStack.shape[0]):
        cs = exp_series[:, i][:, None]  # fingerprint of current slice
        corrs[:, i] = np.sum(cs * pre_series, 0)
        slices[i] = sl_indices[np.argmax(corrs[:, i])]
    return slices, corrs, regions


def MedianPostMatch(preStack, expStack, nRegions=25, radius=5, interpolate=False, nIter=50):
    """
    Performs multiple iterations of PostMatchSlices and returns the median trace
    Args:
        preStack: The stable z pre-stack, assumed to have 21 slices per z-plane
        expStack: The experimental time-stack
        nRegions: The number of regions to be used for matching
        radius: The radius of each region (region itself is square exscribing a circle of this radius)
        interpolate: If true, slice positions will be interpolated 5 fold
        nIter: Number of iterations to perform

    Returns:
        [0]: An array of indices that indicate the most likely position in preStack of each expStack slice
        [1]: The MAD of the slice index trace
    """

    all_slices = np.zeros((nIter, expStack.shape[0]))
    for i in range(nIter):
        s = PostMatchSlices(preStack, expStack, nRegions, radius, interpolate)[0]
        if s is not None:
            all_slices[i, :] = s
        else:
            all_slices[i, :] = np.nan
    return np.nanmedian(all_slices, 0), np.nanmedian(np.abs(np.nanmedian(all_slices, 0)-all_slices), 0)


def ZCorrectTrace(preStack, slice_locations, timeseries, vertices):
    """
    Corrrects the raw fluorescence trace using intensity variations in the preStack according to
    predicted slice locations of each frame
    Args:
        preStack: The pre-stack around the acquisition plane (note slices already summed!!!!)
        slice_locations: Predicted slice locations
        timeseries: The fluorescence timeseries to correct
        vertices: (row,column) tuples of points of the unit graph corresponding to the timeseries

    Returns:
        Corrected version of RawTimeseries attribute
    """
    if slice_locations.size != timeseries.size:
        raise ValueError("Each frame in timeseries needs to have a corresponding slice_location")
    sl_indices = np.arange(preStack.shape[0])
    preValues = np.zeros(preStack.shape[0])
    for v in vertices:
        preValues += preStack[:, v[0], v[1]]
    preValues /= preValues.max()
    f_interp = interp1d(sl_indices, preValues)
    corrector = f_interp(slice_locations)  # np.array([f_interp(s) for s in slice_locations])
    corrector = gaussian_filter(corrector, 1)
    return timeseries / corrector


def MakeNrrdHeader(stack, xy_size, z_size=2.5, unit='"microns"'):
    """
    Creates an Nrrd file header with the corresponding dimension information set
    """
    if stack.ndim != 3:
        raise ValueError('Stack has to be 3D')
    if stack.dtype != np.uint8:
        raise ValueError('Stack has to be of type np.uint8')
    header = dict()
    header['dimension'] = stack.ndim
    header['encoding'] = 'gzip'
    header['keyvaluepairs'] = dict()
    header['kinds'] = ['domain'] * 3
    header['labels'] = ['"x"', '"y"', '"z"']
    header['sizes'] = list(stack.shape)
    header['space dimension'] = 3
    header['space directions'] = [[str(xy_size), '0', '0'],
                                  ['0', str(xy_size), '0'],
                                  ['0', '0', str(z_size)]]
    header['space origin'] = ['0', '0', '0']
    header['space units'] = [unit] * 3
    header['type'] = 'unsigned char'
    return header


def vec_mat_corr(v, m, mean_subtract=True, vnorm=None, mnorm=None):
    """
    Computes the correlation between a vector v and each row
    in the matrix m
    Args:
        v: The vector to correlate to each row in m of size n
        m: A k*n matrix
        mean_subtract: If False assumes that v and m are mean and row-mean subtracted respectively
        vnorm: If not None should be the norm of v
        mnorm: If not None, should be a vector of row-wise vector norms of m

    Returns:
        The correlation coefficients of v to each row in m
    """
    if mean_subtract:
        if v.ndim > 1:
            vsub = v.flatten() - np.mean(v)
        else:
            vsub = v - np.mean(v)
        msub = m - np.mean(m, 1, keepdims=True)
    else:
        if v.ndim > 1:
            vsub = v.flatten()
        else:
            vsub = v
        msub = m
    if vnorm is None:
        vnorm = np.linalg.norm(vsub)
    if mnorm is None:
        mnorm = np.linalg.norm(msub, axis=1)
    return np.dot(vsub, msub.T)/(vnorm*mnorm)
