from collections import deque
import numpy as np
from PIL import Image
import pickle

from collections import Counter
import itertools
import h5py
import matplotlib.pyplot as pl
import matplotlib.path as mpath
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import filters, label, find_objects
from scipy.signal import lfilter
from scipy.interpolate import interp1d

from numba import jit
import numba
from numpy import std, zeros

import cv2

import sys
import os

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


class CellHelper:
    """
    Helper class to keep track of overlapping pixel-to-cell assignments
    """

    pixel_dict = {}  # class variable to keep track of pixel-to-cell assignments across all instances

    @classmethod
    def ResetPixelDict(cls):
        """
        Resets the pixel-to-cell mapping dictionary across all class instances
        """
        cls.pixel_dict = {}

    def __init__(self, x_center, y_center, x_off, y_off, im_width, im_height):
        """
        Creates a new CellHelper instance add all pixels to the pixel-dict as well
        Args:
            x_center: The x-coordinate of the center of this cell
            y_center: The y-coordinate of the center of this cell
            x_off: All x-coordinate offsets to include in this cell
            y_off: All y-coordinate offsets to include in this cell
            im_width: The width of the underlying image for clipping
            im_height: The height of the underlying image for clipping
        """
        self.center = (x_center, y_center)
        self.im_dim = (im_width, im_height)
        self.pixels = [(x, y) for x, y in zip(x_center+x_off, y_center+y_off)
                       if (0 <= x < im_width and 0 <= y < im_height)]
        self.pixels.append(self.center)
        # add the relationship between each of the current points and the current cell to the class dictionary
        for p in self.pixels:
            if p in CellHelper.pixel_dict:
                CellHelper.pixel_dict[p].append(self)
            else:
                CellHelper.pixel_dict[p] = [self]

    def AddPixel(self, x, y):
        """
        Add new point to cell pixels, throw ValueError if already present
        Args:
            x: The pixel's x-coordinate
            y: The pixel's y-coordinate
        """
        p = (x, y)
        if p in self.pixels:
            raise ValueError("Pixel already part of cell")
        self.pixels.append(p)
        if p in CellHelper.pixel_dict:
            CellHelper.pixel_dict[p].append(self)
        else:
            CellHelper.pixel_dict[p] = [self]

    def RemovePixel(self, x, y):
        """
        Remove given point from the cell pixels or raise error if point not part of cell
        Args:
            x: The pixel's x-coordinate
            y: The pixel's y-coordinate
        """
        p = (x, y)
        if p not in self.pixels:
            raise ValueError("Pixel is not part of cell")
        self.pixels.remove(p)
        if p in CellHelper.pixel_dict:
            if self in CellHelper.pixel_dict[p]:
                CellHelper.pixel_dict[p].remove(self)

    def ClearAll(self):
        """
        Clears all pixels and information from this Cell and removes it from the pixel-dict
        """
        self.center = None
        self.im_dim = None
        for p in self.pixels:
            self.RemovePixel(p[0], p[1])
        assert len(self.pixels) == 0
        self.pixels = None

    @property
    def X(self):
        """
        All pixel X-coordinates
        """
        return np.array([p[0]for p in self.pixels])

    @property
    def Y(self):
        """
        All pixel Y-coordinates
        """
        return np.array([p[1] for p in self.pixels])

    @property
    def P_Own(self):
        """
        All points which only belong to this cell
        """
        return [p for p in self.pixels if len(CellHelper.pixel_dict[p]) <= 1]


class CellGraph(GraphBase):
    """
    Performs cell-segmentation of cytoplasmic gcamp labels
    """

    def __init__(self, id, timeseries):
        super().__init__()
        self.V = []
        self.ID = id
        self.RawTimeseries = timeseries

    @staticmethod
    def BreakPixelTies(stack):
        """
        Uses the pixel_dict of cell helper to identify pixels that are assigned to more
        than one cell and subsequently uses time-series correlations to assign these to one cell only
        """
        p_tied = [(k, CellHelper.pixel_dict[k].copy()) for k in CellHelper.pixel_dict.keys()
                  if len(CellHelper.pixel_dict[k]) > 1]
        for p, cells in p_tied:
            max_corr = 0  # maximum correlation of a cell observed so far
            ix_max = -1  # index of cell with maximal correlation
            ts_p = stack[:, p[0], p[1]]
            for i, c in enumerate(cells):
                # build timeseries of the cell under consideration only from pixels that exclusively belong to this cell
                own_pixels = c.P_Own
                ts = np.zeros(stack.shape[0])
                for pix in own_pixels:
                    ts += stack[:, pix[0], pix[1]]
                corr = np.corrcoef(ts_p, ts)[0, 1]
                if corr > max_corr and corr > 0:  # Note: If all corrs<0 pixel gets removed from all cells
                    max_corr = corr
                    ix_max = i
            # second pass across cells - remove pixel in question from all cells that aren't max corr
            for i, c in enumerate(cells):
                if i != ix_max:
                    c.RemovePixel(p[0], p[1])
        # TODO: Remove isolated pixels, i.e. pixels that don't have any 4-connected neighbors belonging to same cell

    @staticmethod
    def GrowCells(cell_list, stack, c_thresh, nPixelsMax):
        """
        For each identified cells tries to grow the border by incorporating new correlated pixels within an
        8-connected neighborhood around current pixels
        Args:
            cell_list: List of CellHelper objects
            stack: Timeseries stack to evaluate pixel-correlations
            c_thresh: Correlation threshold for a pixel to be preliminarily added to the current cell
            nPixelsMax: The maximum number of pixels that can belong to a cell
        """
        def grow_cell(c):
            nonlocal stack, c_thresh, nPixelsMax
            orig_pix = c.pixels.copy()
            new_pix = []
            w, h = c.im_dim
            # establish summed time-series of this cell
            ts = np.zeros(stack.shape[0])
            for pix in orig_pix:
                ts += stack[:, pix[0], pix[1]]
            # keep trying to add pixels as long as there are non-investigated pixels
            while len(orig_pix) > 0 and len(c.pixels) <= nPixelsMax:
                for pix in orig_pix:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            nx = pix[0] + dx
                            ny = pix[1] + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                # only add pixel if it doesn't belong to any other cell yet
                                if (nx, ny) not in CellHelper.pixel_dict:
                                    corr = np.corrcoef(ts, stack[:, nx, ny])[0, 1]
                                    if corr >= c_thresh:
                                        # add new pixel to cell, our new-pixel list and add its timeseries
                                        c.AddPixel(nx, ny)
                                        new_pix.append((nx, ny))
                                        ts += stack[:, nx, ny]
                    if len(c.pixels) > nPixelsMax:
                        # if after finishing the current pixel we exceeded max size finish with this cell
                        return
                # there is no need to re-test for the old pixels but re-do for the newly added pixels
                orig_pix = new_pix.copy()
                new_pix.clear()

        for c in cell_list:
            if len(c.pixels) <= nPixelsMax:
                grow_cell(c)

    @staticmethod
    def RemoveIsolatedPixels(cell_list):
        """
        From each cell in cell_list removes pixels that do not have at least 3 8-connected neighbors
        """
        def n_8_connected(c, p):
            n8 = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = p[0] + dx
                    ny = p[1] + dy
                    if (nx, ny) in c.pixels:
                        n8 += 1
            return n8
        # loop over all cells, removing isolated pixels
        for c in cell_list:
            did_remove = True
            # as long as we removed a pixel in the previous round we need to re-check
            while did_remove:
                did_remove = False
                pixels = c.pixels.copy()
                for px in pixels:
                    n = n_8_connected(c, px)
                    if n < 3:
                        c.RemovePixel(px[0], px[1])
                        did_remove = True

    @staticmethod
    def EnforceAnatomicalSize(cell_list, max_rad):
        """
        For each cell in cell_list removes all pixels that are more than max_rad away from the original cell center
        """
        def center_dist(c, p):
            cx, cy = c.center
            return np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)

        for c in cell_list:
            pixels = c.pixels.copy()
            for px in pixels:
                if center_dist(c, px) > max_rad:
                    c.RemovePixel(px[0], px[1])

    @classmethod
    def CellConnComps(cls, stack, sumImage, cell_diam_px, cell_max_pixels, plot=False):
        """
        Segments a stack of nuclear excluded gcamp based on anatomical as well as correlation-based features
        Args:
            stack: The time-series stack (t,x,y), potentially subsampled along t to save memory
            sumImage: The full-stack sum image across t
            cell_diam_px: The approximate diameter of a cell in pixels
            plot: If True plots segmentation intermediates for diagnostic purposes
            cell_max_pixels: The maximum number of pixels that can be part of one cell

        Returns:
            A list of CellGraphs segmenting the stack
        """
        # 1) use anatomical features in sumImage to determine cell-centroids
        im_scale = gaussian_filter(sumImage, cell_diam_px*10)
        if plot:
            pl.figure()
            pl.imshow(im_scale)
            pl.title("im_scale")
        # compute local minima and maxima via circular structuring element
        d = int(cell_diam_px)
        rad = cell_diam_px / 2
        if d % 2 == 0:
            strel = np.zeros((d-1, d-1))
        else:
            strel = np.zeros((d, d))
        center = strel.shape[0] // 2
        for i in range(strel.shape[0]):
            for j in range(strel.shape[1]):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if dist <= rad:
                    strel[i, j] = 1
        im_min = filters.minimum_filter(sumImage, footprint=strel)
        minima = (sumImage == im_min)
        if plot:
            pl.figure()
            pl.imshow(minima)
            pl.title("minima")
        im_max = filters.maximum_filter(sumImage, footprint=strel)
        # compute difference to later threshold minima
        im_diff = (im_max - im_min) / im_scale
        threshold = 0.2  # np.percentile(im_diff.ravel(), 15)
        if plot:
            pl.figure()
            vals = pl.hist(im_diff.ravel(), 250)[0]
            pl.plot([threshold, threshold], [0, vals.max()], 'r')
            pl.title("Histogram of max-min differences")
        # threshold minima and extract centroids of connected components
        minima[im_diff < threshold] = 0
        if plot:
            pl.figure()
            pl.imshow(minima)
            pl.title("minima - thresholded")
        labeled, num_objects = label(minima)
        slices = find_objects(labeled)
        # create coordinate arrays of the centers of the detected components
        x, y = [], []
        for dx, dy in slices:
            x_center = (dx.start + dx.stop - 1)/2
            y_center = (dy.start + dy.stop - 1)/2
            # append only centroids that are at least cell radius away from each image edge
            if rad < x_center < sumImage.shape[0]-rad and rad < y_center < sumImage.shape[1]-rad:
                x.append(x_center)
                y.append(y_center)
        x = np.array(x).astype(int)
        y = np.array(y).astype(int)
        if plot:
            log_image = np.log(sumImage)
            log_image[np.isinf(log_image)] = 0
            log_image = log_image / log_image.max() * (2**16)
            log_image = log_image.astype(np.uint16)
            pl.figure()
            pl.imshow(log_image, cmap="bone")
            pl.plot(y, x, 'r.')
            # plot centroid triggered average view
            view_size = 4*d + 1
            cent_view = np.zeros((view_size, view_size))
            for xp, yp in zip(x, y):
                if 2*d < xp < sumImage.shape[0]-2*d and 2*d < yp < sumImage.shape[1]-2*d:
                    sub_view = sumImage[xp-2*d:xp+2*d+1, yp-2*d:yp+2*d+1] / im_scale[xp-2*d:xp+2*d+1, yp-2*d:yp+2*d+1]
                    sub_view -= np.mean(sub_view)
                    sub_view /= np.std(sub_view)
                    cent_view += sub_view
            pl.figure()
            pl.imshow(cent_view / xp.size)
            pl.title("Average surround of detected centroids")
        # perform preliminary assignment of pixels to cells based on cell radius
        x_offsets, y_offsets = [], []
        for i in range(int(rad) + 2):
            for j in range(int(rad) + 2):
                if 0 < np.sqrt(i**2 + j**2) <= rad:
                    x_offsets.append(i)
                    y_offsets.append(j)
                    if i > 0:
                        x_offsets.append(-i)
                        y_offsets.append(j)
                    if j > 0:
                        x_offsets.append(i)
                        y_offsets.append(-j)
                    if i > 0 and j > 0:
                        x_offsets.append(-i)
                        y_offsets.append(-j)
        x_offsets = np.array(x_offsets).astype(int)
        y_offsets = np.array(y_offsets).astype(int)
        # build list of potential cells
        CellHelper.ResetPixelDict()
        cell_list = []
        for px, py in zip(x, y):
            cell_list.append(CellHelper(px, py, x_offsets, y_offsets, sumImage.shape[0], sumImage.shape[1]))
        if plot:
            pl.figure()
            pl.imshow(log_image, cmap='bone')
            for c in cell_list:
                pl.plot(c.Y, c.X, '.', alpha=0.5)
            pl.title("Original cell territories")

        # 2) Use timeseries correlation to break pixel-ties (assign multi-assigned pixels to one cell only)
        cls.BreakPixelTies(stack)
        if plot:
            pl.figure()
            pl.imshow(log_image, cmap='bone')
            for c in cell_list:
                pl.plot(c.Y, c.X, '.', alpha=0.5)
            pl.title("Cell territories after first tie-break")

        # 3) Go through cycles of growth - the following threshold values are based on the observation that
        # without spatial smoothing the maximal correlation btw. timeseries is around 0.6 and the smallest
        # significant correlation is around 0.007
        for thresh in np.linspace(0.7, 0.01, 15):
            cls.GrowCells(cell_list, stack, thresh, cell_max_pixels)
        # our growth operation should not have lead to overlapping pixels
        assert max([len(CellHelper.pixel_dict[k]) for k in CellHelper.pixel_dict.keys()]) < 2

        cls.RemoveIsolatedPixels(cell_list)
        cls.EnforceAnatomicalSize(cell_list, rad*4/3)

        if plot:
            pl.figure()
            pl.imshow(log_image, cmap='bone')
            for c in cell_list:
                pl.plot(c.Y, c.X, '.', alpha=0.5)
            pl.title("Final territories")
            pl.figure()
            pl.hist([len(c.pixels) for c in cell_list], 25)
            pl.title("Cell size distribution")

        # 4) Transform the cell list into a list of connected components
        graph_list = []
        for i, c in enumerate(cell_list):
            ts = np.zeros(stack.shape[0], np.float32)
            g = CellGraph(i, ts)
            for p in c.pixels:
                g.V.append((p[0], p[1], 0))  # In our other graphs, the third component identifies the BFS generation
                g.RawTimeseries += stack[:, p[0], p[1]]
            graph_list.append(g)
        return graph_list


class CorrelationGraph(GraphBase):

    def __init__(self, id, timeseries):
        super().__init__()
        self.V = []  # list of vertices
        self.ID = id  # id of this graph for easy component reference
        self.Timeseries = timeseries  # the summed pixel-timeseries of the graph

    @staticmethod
    def CorrelationConnComps(stack, im_ncorr, corr_thresh, predicate, norm8=True, limit=None, seed_limit=0, maxsize=np.inf):
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
            maxsize: Maximal allowed size of each graph
            RETURNS:
                [0]: List of connected component graphs
                [1]: Image numerically identifying each pixel of each graph
        """
        def BFS(stack, thresh, visited, sourceX, sourceY, color, norm8, predicate, maxsize):
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
                        if c >= thresh and len(pg.V) <= maxsize:
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
            conn_comps.append(BFS(stack, corr_thresh, visited, x, y, curr_color, norm8, predicate, maxsize))
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
        # the following can optionally be assigned later
        self.graph_info = []  # for each unit (row) in RawData file and anatomical location of its origin
        self.original_time_per_frame = 0  # acquisition tim of each microscope frame before interpolation

    @property
    def NFrames(self):
        return self.RawData.shape[1]

    @property
    def NUnits(self):
        return self.RawData.shape[0]

    # @property
    # def AvgUnitBrightness(self):
    #     """
    #     The average brightness of each unit in the experiment
    #     """
    #     if len(self.graph_info) == 0:
    #         return None
    #     avg_brightness = np.zeros(len(self.graph_info), dtype=np.float32)
    #     projection_dict = {}
    #     for i, gi in enumerate(self.graph_info):
    #         sf = gi[0]
    #         if sf in projection_dict:
    #             projection = projection_dict[sf]
    #         else:
    #             # first attempt to load aligned stack
    #             stackFile = sf[:-4]+"_stack.npy"
    #             if os.path.exists(stackFile):
    #                 stack = np.load(stackFile).astype(np.float32)
    #             else:
    #                 stack = OpenStack(sf).astype(np.float32)
    #             projection = np.sum(stack, 0)
    #             projection_dict[sf] = projection
    #         bsum = 0
    #         for v in gi[1]:
    #             bsum += projection[v[0], v[1]]
    #         avg_brightness[i] = bsum / len(gi[1])
    #     return avg_brightness

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

    def get_raw_unit_traces(self, indices):
        """
        Compute summed activity traces for given graphs in the original aligned stack interpolated appropriately
        Args:
            indices: List of indices for which to return traces

        Returns:
            len(indices) x n_timepoints matrix of original traces (interpolated to the frequency of the dataset)
        """
        if self.original_time_per_frame == 0:
            raise ValueError("Cannot interpolate if original_time_per_frame is not set on class")
        raw = []
        # for efficiency (as long as indices are ordered) we store the last accessed stack since it is
        # probably that the next index refers to a unit in the same slice
        last_name = ""
        last_stack = None
        try:
            for i in indices:
                sf = self.graph_info[i][0]
                if sf == last_name:
                    stack = last_stack
                else:
                    # first attempt to load aligned stack
                    stackFile = sf[:-4] + "_stack.npy"
                    if os.path.exists(stackFile):
                        stack = np.load(stackFile).astype(np.float32)
                    else:
                        stack = OpenStack(sf).astype(np.float32)
                last_name = sf
                last_stack = stack
                bsum = np.zeros((1, stack.shape[0]), dtype=np.float32)
                for v in self.graph_info[i][1]:
                    bsum += stack[:, v[0], v[1]]
                raw.append(bsum)
        except TypeError:
            print("indices argument should be iterable attribute")
            raise
        lens = [r.size for r in raw]
        ml = min(lens)
        raw = np.vstack([r[0, :ml] for r in raw])
        # interpolate raw data
        nframes_interp = (self.preFrames + self.stimFrames + self.postFrames) * self.nRepeats
        interp_times = np.arange(nframes_interp) / self.frameRate
        frame_times = np.arange(raw.shape[1]) * self.original_time_per_frame
        ipol = lambda y: np.interp(interp_times, frame_times, y[:frame_times.size])
        return np.vstack([ipol(row) for row in raw])


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


class DetailCharExperiment(ImagingData):
    """
    Class to describe our fixed-protocol detailed characterization experiment
    """
    def __init__(self, imagingData, swimVigor, n_repeats, t_p_f_orig, graph_info, caTimeConstant=0.4, frameRate=5):
        """
        Creates a new DetailCharExperiment
        Args:
            imagingData: The (interpolated) calcium imaging data
            swimVigor: The swim vigor at same frequency as imagingData
            n_repeats: The number of repeats per plane
            t_p_f_orig: The original time per frame during acquisition
            graph_info: Graph info for each cell
            caTimeConstant: The indicator time constant
            frameRate: Interpolated frame-rate of imagingData and swimVigor
        """
        super().__init__(imagingData, swimVigor)
        self.n_repeats = n_repeats
        self.original_time_per_frame = t_p_f_orig
        self.frameRate = frameRate
        self.caTimeConstant = caTimeConstant
        self.graph_info = graph_info
        self.totalSeconds = self.RawData.shape[1] / self.frameRate

    def repeat_align(self, activity_mat: np.ndarray) -> np.ndarray:
        """
        Aligns the given matrix by repeats
        Args:
            activity_mat: The matrix or vector to align by repeats

        Returns:
            n_cells x n_repeats x n_timepoints array of repeat aligned data
        """
        if activity_mat.ndim > 2:
            raise ValueError("activity_mat can't have more than two dimensions")
        elif activity_mat.ndim == 1:
            amat = activity_mat.reshape((self.n_repeats, activity_mat.size // self.n_repeats))
            amat = amat[None, :, :]
        else:
            amat = activity_mat.reshape((activity_mat.shape[0],
                                         self.n_repeats, activity_mat.shape[1] // self.n_repeats))
        return amat

    def variance_score(self) -> np.ndarray:
        """
        For each cell computes the activity variance score as var(average)/avg(repeat_vars)
        """
        amat = self.repeat_align(self.RawData)
        # for each cell compute the variance of the repeat average
        rep_avgs = np.mean(amat, 1)  # n_cells * repeat_time
        var_of_avg = np.var(rep_avgs, 1)  # n_cells
        # for each cell compute the average variance across individual repeats
        var_per_rep = np.var(amat, 2)  # n_cells * n_repeats
        avg_of_vars = np.mean(var_per_rep, 1)
        return var_of_avg / avg_of_vars

    def get_sine_pre_stim_ix(self):
        """
        Return pre and stim indices for sinewave period
        """
        frames = np.arange(self.totalSeconds // self.n_repeats * self.frameRate, dtype=int)
        time = frames / self.frameRate
        pre_frames = frames[np.logical_and(time >= 78, time <= 82)]
        stim_frames = frames[np.logical_and(time >= 90, time <= 110)]
        return pre_frames, stim_frames

    def get_lstep_pre_stim_ix(self):
        """
        Return pre and stim indices for 1500mW step
        """
        frames = np.arange(self.totalSeconds // self.n_repeats * self.frameRate, dtype=int)
        time = frames / self.frameRate
        pre_frames = frames[np.logical_and(time >= 35, time <= 40)]
        stim_frames = frames[np.logical_and(time >= 43, time <= 46)]
        return pre_frames, stim_frames

    def get_tap_pre_stim_ix(self):
        """
        Retunr pre and stim indices for tap
        """
        frames = np.arange(self.totalSeconds // self.n_repeats * self.frameRate, dtype=int)
        time = frames / self.frameRate
        pre_frames = frames[np.logical_and(time >= 122, time <= 128)]
        stim_frames = frames[np.logical_and(time > 128, time <= 133)]
        return pre_frames, stim_frames

    def activations(self, ix_pre_repeat, ix_stim_repeat):
        """
        Compute average activity for all traces per repeat in activity_mat over given pre and stim indices
        Args:
            ix_pre_repeat: Indices in repeat aligned trace that identify pre-period
            ix_stim_repeat: Indices in repeat aligned trace that identify stim-period

        Returns:
            [0]: n_cells x n_repeat matrix of pre averages
            [1]: n_cells x n_repeat matrix of stim averages
        """
        rep_aligned = self.repeat_align(self.RawData)
        pre = np.mean(rep_aligned[:, :, ix_pre_repeat], 2)
        stim = np.mean(rep_aligned[:, :, ix_stim_repeat], 2)
        return pre, stim

    def fold_activations(self, ix_pre_repeat, ix_stim_repeat):
        """
        Compute per-repeat z-scored change in activity in relation to pre-standard deviations
        Args:
            ix_pre_repeat: Indices in repeat aligned trace that identify pre-period
            ix_stim_repeat: Indices in repeat aligned trace that identify stim-period

        Returns:
            n_cells x n_repeat array of z-scores
        """
        rep_aligned = self.repeat_align(self.RawData)
        # estimate noise using pre-periods
        pre_std = np.std(rep_aligned[:, :, ix_pre_repeat], 2, keepdims=True) + 1e-3
        pre_avg = np.mean(rep_aligned[:, :, ix_pre_repeat], 2, keepdims=True)
        stim = np.mean(rep_aligned[:, :, ix_stim_repeat], 2, keepdims=True)
        return ((stim - pre_avg) / pre_std)[:, :, 0]  # remove last dimension now


class MotorContainer:
    """
    Class to memory efficiently store per-cell motor events across multiple experiments by only storing one trace per
    plane and using an overridden indexer to make it indexable like a full matrix
    """

    def __init__(self, sourceFiles, final_timebase, ca_time_constant, predicate=None, tdd=None, hdf5_store=None):
        """
        Creates a new MotorContainer
        Args:
            sourceFiles: For each cell / unit the source-file name to obtain motor traces or a list of tuples of
             (sourceFile, ca_time_base)
            final_timebase: The timebase to which the motor trace should be binned
            ca_time_constant: The calcium indicator time-constant for convolution
            predicate: Used on each loaded TailData object to decide which bout starts should be included - this
             function should take a TailData object as argument and return a bout-start trace
            tdd: Optionally a pre-filled TailDataDict to avoid loading tail files multiple times when generating many
             containers
            hdf5_store: Optionally when creating a tail data dict an hdf5 file for efficient storage of tail data
        """
        self.sourceFiles = sourceFiles
        self.final_timebase = final_timebase
        self.ca_time_constant = ca_time_constant
        self.predicate = predicate
        self.traces = {}
        if tdd is None:
            tdd = TailDataDict(ca_time_constant, hdf5_store=hdf5_store)
        self.tdd = tdd

    def bin_trace(self, trace, frameTimes):
        """
        Bins a starting trace according to the timebase
        Args:
            trace: The trace to bin
            frameTimes: The acquisition time for each frame in trace

        Returns:
            trace binned according to self.final_timebase
        """
        digitized = np.digitize(frameTimes, self.final_timebase)
        return np.array([trace[digitized == i].sum() for i in range(1, self.final_timebase.size)])

    @property
    def avg_motor_output(self):
        """
        Returns the per-plane average motor output
        """
        # make sure that our traces dictionary contains all necessary traces
        for sf in self.sourceFiles:
            if sf not in self.traces:
                self._add_trace(sf)
        return np.mean(np.vstack([v for v in self.traces.values()]), 0)

    def _get_row(self, ix):
        """
        Returns the start trace that belongs to the given cell's index
        """
        sf = self.sourceFiles[ix]
        if sf in self.traces:
            return self.traces[sf]
        else:
            return self._add_trace(sf)

    def _add_trace(self, sf):
        """
        Adds a new trace to the dictionary and returns it
        """
        tdata = self.tdd[sf]
        if self.predicate is None:
            start_trace = tdata.starting
        else:
            start_trace = self.predicate(tdata)
        # convolve the trace
        start_trace = CaConvolve(start_trace, self.ca_time_constant, 100)
        start_trace = self.bin_trace(start_trace, tdata.frameTime)
        self.traces[sf] = start_trace
        return start_trace

    def _get_rows(self, r_ix):
        """
        Returns full rows at the given index or slice
        Args:
            r_ix: The row index or slice

        Returns:
            A vector/matrix of the indicated row(s)
        """
        if np.issubdtype(type(r_ix), np.int):
            return self._get_row(r_ix)
        start, stop, step = r_ix.indices(len(self.sourceFiles))
        rows = [self._get_row(i) for i in range(start, stop, step)]
        if len(rows) == 1:
            # slice only referenced a single row
            return np.array(rows)
        else:
            return np.vstack(rows)

    def _get_rows_cols(self, r_ix, c_ix):
        row_all = self._get_rows(r_ix)
        if row_all.ndim == 1:
            return row_all[c_ix]
        else:
            return row_all[:, c_ix]

    def __getitem__(self, item):
        if type(item) == tuple:
            if len(item) > 2:
                raise ValueError("Indexer dimensionality does not match data")
            for i in item:
                if not np.issubdtype(type(i), np.int) and type(i) != slice:
                    raise TypeError("Only integers and slice objects accepted as indices")
        elif not np.issubdtype(type(item), np.int) and type(item) != slice:
            raise TypeError("Only integers and slice objects accepted as indices")
        if type(item) != tuple:
            # we only got a single index expression - interpret as row index
            return self._get_rows(item)
        return self._get_rows_cols(item[0], item[1])

    def __iter__(self):
        ix = 0
        while ix < len(self.sourceFiles):
            yield self._get_row(ix)
            ix += 1


class PixelGraph:

    def __init__(self, id):
        self.V = []  # list of vertices
        self.ID = id  # id of this graph for easy component reference

    @property
    def NPixels(self):
        return len(self.V)
# class PixelGraph


# compiled acceleration


@jit(numba.float64[:](numba.float64[:], numba.int32))
def Vigor(cumAngle, winlen=10):
    """
    Computes the swim vigor based on a cumulative angle trace
    as the windowed standard deviation of the cumAngles
    """
    s = cumAngle.size
    vig = zeros(s)
    for i in range(winlen, s):
        vig[i] = std(cumAngle[i-winlen+1:i+1])
    return vig


class TailData:

    def __init__(self, fileData, ca_timeconstant, frameRate, scan_frame_length=None):
        """
        Creates a new TailData object
        Args:
            fileData: Matrix loaded from tailfile
            ca_timeconstant: Timeconstant of calcium indicator used during experiments
            frameRate: The tail camera acquisition framerate
            scan_frame_length: For more accurate alignment the time it took to acquire each 2P scan frame
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
        self.velcty_noise = np.nanstd(self.velocity[self.velocity < 4])
        # compute a time-trace assuming a constant frame-rate which starts at 0
        # for the likely first camera frame during the first acquisition frame
        # we infer this frame by going back avgCount frames from the first frame
        # of the second (!) scan frame (i.e. this should then be the first frame
        # of the first scan frame)
        frames = np.arange(self.cumAngles.size)
        first_frame = np.min(frames[self.scanFrame == 1]) - avgCount.astype(int)
        # remove overhang from frame 0 call
        self.scanFrame[:first_frame] = -1
        if scan_frame_length is not None:
            # build frame-time tied to the scan-frame clock in case camera acquisition does not follow the
            # intended frame rate: For the camera frame which is in the middle of a given scan-frame set
            # its time to the middle time of scan acquisition of that frame. Then interpolate times in between
            ix_key_frame = []
            key_frame_times = []
            # add very first frame and its approximated time purely based on camera time
            ix_key_frame.append(0)
            key_frame_times.append((frames[0] - first_frame) / frameRate)
            for i in range(self.scanFrame.max()):
                ix_key = int(np.mean(frames[self.scanFrame == i]))
                key_time = scan_frame_length / 2 + scan_frame_length * i
                ix_key_frame.append(ix_key)
                key_frame_times.append(key_time)
            # use linear interpolation to create times for each frame
            self.frameTime = np.interp(frames, np.array(ix_key_frame), np.array(key_frame_times), right=np.nan)
            self.frameTime = self.frameTime[np.logical_not(np.isnan(self.frameTime))]
        else:
            frames -= first_frame
            self.frameTime = (frames / frameRate).astype(np.float32)
        # create bout-start trace at original frame-rate
        self.starting = np.zeros_like(self.frameTime)
        if self.bouts is not None:
            bout_starts = self.bouts[:, 0].astype(int)
            # since we potentially clip our starting trace to the last valid frame-time (experiment end)
            # we also only include bout-starts that occured up to that index
            bout_starts = bout_starts[bout_starts < self.frameTime.size]
            self.starting[bout_starts] = 1

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
    def LoadTailData(filename, ca_timeConstant, frameRate=100, scan_frame_length=None):
        try:
            data = np.genfromtxt(filename, delimiter='\t')
        except (IOError, OSError):
            return None
        return TailData(data, ca_timeConstant, frameRate, scan_frame_length)

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
    """
    Class for managing TailData objects across multiple experiments
    without needing to re-read files from disk
    """
    def __init__(self, ca_timeConstant=1.796, frameRate=100, hdf5_store=None):
        """
        Creates a new TailDataDict
        Args:
            ca_timeConstant: Indicator time-constant for convolution
            frameRate: Frame-rate of original tail data
            hdf5_store: Optional handle of hdf5 file which will be probed for storing pickled TailData objects
        """
        self._td_dict = dict()
        self.ca_timeConstant = ca_timeConstant
        self.frameRate = frameRate
        self.hdf5_store = hdf5_store

    def __getitem__(self, item):
        """
        Indexer into our TailData dictionary
        Args:
            item: Either a string identifying the underlying tail file or a tuple (string, float) identifying the
             underlying tail file as well as the original calcium acquisition frame-rate for alignment

        Returns:
            A TailData object corresponding to the file
        """
        if not np.issubdtype(type(item), np.str):
            if type(item) != tuple or len(item) != 2 or not np.issubdtype(type(item[0]), np.str) \
                    or not np.issubdtype(type(item[1]), np.float):
                raise TypeError('Indexer needs to be string with filename or tuple (string, float)')
        if type(item) == tuple:
            scan_frame_length = item[1]
            item = item[0]
        else:
            scan_frame_length = None
        if len(item) > 4 and item[-4:] == 'tail':
            tf = item
        else:
            tf = self.tailFile(item)
        if tf in self._td_dict:
            return self._td_dict[tf]
        else:
            if self.hdf5_store is not None and tf in self.hdf5_store:
                # try more efficient unpickling
                td = pickle.loads(np.array(self.hdf5_store[tf]))
            else:
                try:
                    td = TailData.LoadTailData(tf, self.ca_timeConstant, self.frameRate, scan_frame_length)
                except IOError:
                    raise KeyError('Could not find taildata for file {0}'.format(tf))
            if td is None:
                raise KeyError('Could not find taildata for file {0}'.format(tf))
            self._td_dict[tf] = td
            return td

    def __contains__(self, item):
        if type(item) == tuple:
            item = item[0]
        if len(item) > 4 and item[-4:] == 'tail':
            tf = item
        else:
            tf = self.tailFile(item)
        return tf in self._td_dict

    @staticmethod
    def tailFile(tif_name):
        return tif_name[:-6]+'.tail'

    @property
    def fileNames(self):
        return self._td_dict.keys()
# TailDataDict


class KDNode:
    """
    Node of a k-d tree
    """
    def __init__(self, location: np.ndarray, axis, parent=None, left=None, right=None):
        """
        Creates a new KDNode
        Args:
            location: The point location of this node
            axis: The coordinate axis along which the split occurs (p[axis]<location[axis]->left, otherwise right)
            parent: The node's parent
        """
        self.l = left
        self.r = right
        self.p = parent
        self.location = location
        self.axis = axis


class KDTree:
    """
    Implements a k-d tree
    """
    def __init__(self, points: np.ndarray, copy_points=True, split_by_spread=False):
        """
        Creates a new k-d tree
        Args:
            points: The points to bulk-insert into the tree
            copy_points: If true points will be copied before insertion, otherwise points may be changed
            split_by_spread: If true coordinate splits won't simply alternate but will occur along axis with largest est. spread
        """
        self.size = points.shape[0]
        if copy_points:
            pts = points.copy()
        else:
            pts = points
        if pts.ndim == 1:
            pts = pts[:, None]
        self.k = pts.shape[1]
        self.sbs = False
        if self.k > 1:
            # only consider doing spread calculations if there is more than one dimension
            self.sbs = split_by_spread
        self.root = self._bulk_insert(pts)
    
    @staticmethod
    def _largest_spread(points: np.ndarray) -> int:
        """
        Computes the dimension of largest spread of points
        Args:
            points: For which to compute spread along each dimension

        Returns:
            The dimension 0 <= d < points.shape[1] with the largest coordinate spread
        """
        top = np.percentile(points, 95, 0, interpolation="higher")
        bottom = np.percentile(points, 5, 0, interpolation="lower")
        return np.argmax(top - bottom)

    def _bulk_insert(self, points: np.ndarray, parent=None, depth=0) -> KDNode:
        """
        Creates new rooted k-d tree by recursively inserting the points
        Args:
            points: The points to insert
            depth: The depth of the insertion (to determine sorting axis)
            parent: This subtree's parent

        Returns:
            A node with left and right subtrees containing the points
        """
        if points.size == 0:
            return None
        # calculate spread if requested but never for small point clouds
        if self.sbs and points.shape[0] > 10:
            axis = self._largest_spread(points)
        else:
            axis = depth % self.k
        median = points.shape[0] // 2
        # sort points along axis to find median point
        # NOTE: In theory, np.argpartition should be faster at least for large numbers of points, since its order
        # of growth is O(n) compared to O(n log n) of full sort. However, even for 1e6 points (which is probably
        # the max we'll ever see) sorting is still in fact faster for building the tree. This is likely due to the
        # fact that most calls to sort or partition (due to the tree structure) will be made for very small amounts
        # of data, for which np.argsort is in fact faster (still slower for n=1000 but faster for n=100 (or below)
        points = points[np.argsort(points[:, axis]), :]
        N = KDNode(points[median, :].copy(), axis, parent)
        N.l = self._bulk_insert(points[:median, :], N, axis+1)
        N.r = self._bulk_insert(points[median+1:, :], N, axis+1)
        return N

    def insert(self, point):
        """
        Inserts a new point into the k-d tree
        """
        N = KDNode(point, -1)
        N.l = None
        N.r = None
        if self.root is None:
            N.axis = 0
            self.root = N
            N.p = None
        else:
            node = self.root
            while True:
                i = node.axis
                if N.location[i] < node.location[i]:
                    if node.l is None:
                        N.axis = (i + 1) % self.k  # switch axis
                        N.p = node
                        node.l = N
                        return
                    node = node.l
                else:
                    if node.r is None:
                        N.axis = (i + 1) % self.k
                        N.p = node
                        node.r = N
                        return
                    node = node.r

    def nearest_neighbor(self, point: np.ndarray, allow_0=True):
        """
        Find a point's nearest neighbor it's distance
        Args:
            point: The point for which to find the nearest neighbor
            allow_0: If set to false, 0-distance points won't be returned and next nearest will be chosen

        Returns:
            [0]: The nearest neighbor coordinates
            [1]: The euclidian distance to the nearest neighbor
        """
        def nn(node: KDNode, p: np.ndarray, allow_0):
            """
            Recursively finds the nearest neighbor and distance of p
            """
            # algorithm: Go down from node until node is None. On the way back up
            # assign nearest neighbor. Whenever the distance along the split axis
            # for a node is smaller than the current best distance it means that
            # there can theoretically be nodes on the other side of the split
            # which are closer than the current best neightborh. Hence that subtree
            # has to be searched as well
            if node is None:
                return None, np.inf

            # calculate current node's distance along split axis
            d_axis = (p[node.axis]-node.location[node.axis])**2  # NOTE: Don't adjust d_axis based on allow_0

            if p[node.axis] < node.location[node.axis]:
                # search on left
                loc, best = nn(node.l, p, allow_0)
                # check if we need to test the other (r) side of the  and whether to update best with current node
                if d_axis <= best:
                    d_current = np.sum((node.location - p) ** 2)
                    if not allow_0 and d_current == 0:
                        pass
                    elif d_current < best:
                        best = d_current
                        loc = node.location
                    loc2, best2 = nn(node.r, p, allow_0)
                    if best2 < best:
                        best = best2
                        loc = loc2
                return loc, best
            else:
                loc, best = nn(node.r, p, allow_0)
                # check if we need to test the other (l) side of the tree
                if d_axis <= best:
                    d_current = np.sum((node.location - p) ** 2)
                    if not allow_0 and d_current == 0:
                        pass
                    elif d_current < best:
                        best = d_current
                        loc = node.location
                    loc2, best2 = nn(node.l, p, allow_0)
                    if best2 < best:
                        best = best2
                        loc = loc2
                return loc, best

        if type(point) is not np.ndarray:
            # try to fix this mess
            point = np.array([point])
        point, d = nn(self.root, point, allow_0)
        return point, np.sqrt(d)

    def nearest_n_neighbors(self, point: np.ndarray, n, allow_0=True):
        """
        Find a points n nearest neighbors in a tree
        Args:
            point: The point for which to find the nearest neighbors
            n: The number of neighbors to return
            allow_0: Whether to consider a 0 distance a valid find (likely identity as opposed to neighbor)

        Returns:
            [0]: The n coordinates of the nearest neighbors
            [1]: The euclidian distances to all neighbors
        """
        def insert_sorted(d, new_d, points, new_point):
            """
            If new_d is smaller than any element of d, inserts new_d into d such that d always remains in ascending
            sorted order. Mirrors all operations on d with the points array.
            """
            if new_d >= d[-1]:
                # no update necessary
                return
            for i in range(d.size):
                if new_d < d[i]:
                    b = d[i]
                    d[i] = new_d
                    new_d = b
                    b = points[i, :].copy()
                    points[i, :] = new_point
                    new_point = b

        def nnn(node: KDNode, p: np.ndarray, n, allow_0):
            """
            Recursively find the n nearest neighbors of p and their distances
            """
            if node is None:
                return np.zeros((n, self.k)), np.full(n, np.inf)

            d_axis = (p[node.axis] - node.location[node.axis]) ** 2
            if p[node.axis] < node.location[node.axis]:
                # search on left
                loc, best = nnn(node.l, p, n, allow_0)
                # check if we need to test the other (r) side of the tree and if the current node itself is a contender
                if d_axis <= best[-1]:
                    # calculate current node's full distance
                    d_current = np.sum((node.location - p) ** 2)
                    if allow_0 or d_current != 0:
                        insert_sorted(best, d_current, loc, node.location)
                    loc2, best2 = nnn(node.r, p, n, allow_0)
                    for i in range(n):
                        insert_sorted(best, best2[i], loc, loc2[i, :])
                return loc, best
            else:
                loc, best = nnn(node.r, p, n, allow_0)
                # check if we need to test the other (l) side of the tree and if the current node itself is a contender
                if d_axis <= best[-1]:
                    d_current = np.sum((node.location - p) ** 2)
                    if allow_0 or d_current != 0:
                        insert_sorted(best, d_current, loc, node.location)
                    loc2, best2 = nnn(node.l, p, n, allow_0)
                    for i in range(n):
                        insert_sorted(best, best2[i], loc, loc2[i, :])
                return loc, best

        if type(point) is not np.ndarray:
            # try to fix this mess
            point = np.array([point])
        if n < 1:
            raise ValueError("n has to be at least 1")
        if n > self.size:
            raise ValueError("Can't search for more neighbors than elements in the tree")
        point, d = nnn(self.root, point, n, allow_0)
        return point, np.sqrt(d)

    def count_neighbors(self, point, radius):
        """
        Counts the number of points in this tree that are within a hypersphere
        Args:
            point: The center of the sphere
            radius: The spheres radius (maximum distance)

        Returns:
            Count of tree points within the sphere
        """
        def cnt(node: KDNode, p: np.ndarray, r_squared):
            """
            Recursively counts all neighbors around p within radius distance
            """
            if node is None:
                return

            nonlocal count

            # calculate current node's distance along split axis and full distance if within reach along axis
            d_axis = (p[node.axis]-node.location[node.axis])**2

            if d_axis < r_squared and np.sum((node.location - p) ** 2) < r_squared:
                count += 1

            if p[node.axis] < node.location[node.axis]:
                # count on left
                cnt(node.l, p, r_squared)
                # check if we need to test the other (r) side of the tree
                if d_axis <= r_squared:
                    cnt(node.r, p, r_squared)
                return
            else:
                cnt(node.r, p, r_squared)
                # check if we need to test the other (l) side of the tree
                if d_axis <= r_squared:
                    cnt(node.l, p, r_squared)
                return
        if type(point) is not np.ndarray:
            point = np.array([point])
        count = 0
        cnt(self.root, point, radius**2)
        return count

    def min_distances(self, points: np.ndarray, allow_0=False):
        """
        For each point in points returns the distance to its closest neighbor
        Args:
            points: The points for which to find the closest neighbor distance
            allow_0: Determines whether points with the exact same location are considered neighbors or not

        Returns:
            The minimum distance for each point in points
        """
        if points.ndim == 1:
            points = points[:, None]
        if points.shape[1] != self.k:
            raise ValueError("Dimensionality (k) of tree and points array does not match")
        md = np.empty(points.shape[0])
        for i, p in enumerate(points):
            md[i] = self.nearest_neighbor(p, allow_0)[1]
        return md

    def avg_min_distances(self, points: np.array, n_neighbors, allow_0=False):
        """
        For each point in points returns the average distance to its n closest neighbors
        Args:
            points: The points for which to find the distances
            n_neighbors: The number of closest neighbors to include
            allow_0: Determines whether points with the exact same location are considered neighbors or not

        Returns:
            The average distance for each point in points to its n closest neighbors
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors hast to be at least 1")
        if n_neighbors == 1:
            return self.min_distances(points, allow_0)
        if points.ndim == 1:
            points = points[:, None]
        if points.shape[1] != self.k:
            raise ValueError("Dimensionality (k) of tree and points array does not match")
        md = np.empty(points.shape[0])
        for i, p in enumerate(points):
            md[i] = np.mean(self.nearest_n_neighbors(p, n_neighbors, allow_0)[1])
        return md


class RegionContainer:
    """
    Container for saving and loading RegionROI information
    """

    def __init__(self, positions, region_name: str, z_index: int):
        """
        Create a new RegionContainer
        :param positions: The polygon vertices of the ROI
        :param region_name: The name of this region
        :param z_index: The index of the z-plane that this ROI came from
        """
        self.positions = positions
        self.region_name = region_name
        self.z_index = z_index

    def point_in_region(self, point):
        """
        Tests whether a point is within the region or outside
        :param point: The test point
        :return: True if the point is within the region False if on a boundary or outside
        """
        # even when setting closed to True we still need to supply the first point twice
        poly_path = mpath.Path(self.positions + [self.positions[0]], closed=True)
        return poly_path.contains_point(point)

    @staticmethod
    def save_container_list(container_list, dfile: h5py.File):
        """
        Saves a list of RegionContainer objects to an hdf5 file
        :param container_list: The list of RegionContainer objects to save
        :param dfile: Handle of hdf5 file to which the list should be saved
        """
        key = 0
        for rc in container_list:
            while str(key) in dfile:
                key += 1
            dfile.create_group(str(key))
            dfile[str(key)].create_dataset(name="positions", data=np.vstack(rc.positions))
            dfile[str(key)].create_dataset(name="region_name", data=rc.region_name)
            dfile[str(key)].create_dataset(name="z_index", data=rc.z_index)

    @staticmethod
    def load_container_list(dfile: h5py.File):
        """
        Loads a list of RegionContainer objects from an hdf5 file
        :param dfile: Handle of hdf5 file from which list should be loaded
        :return: A list of RegionContainer objects
        """
        container_list = []
        for k in dfile.keys():
            try:
                pos = np.array(dfile[k]["positions"])
                pos = [(p[0], p[1]) for p in pos]
                rn = str(np.array(dfile[k]["region_name"]))
                zi = int(np.array(dfile[k]["z_index"]))
                rc = RegionContainer(pos, rn, zi)
                container_list.append(rc)
            except KeyError:
                warnings.warn("Found non RegionContainer object in file {0}".format(dfile.filename))
                continue
        return container_list


def UiGetFile(filetypes=[('Tiff stack', '.tif;.tiff')], multiple=False):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    options = {'filetypes': filetypes, 'multiple': multiple}
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
    if ca_timeconstant == 0:
        return trace
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
                        trace += 0.5 * stack[:, x, y]
                    else:
                        trace += 1/16 * stack[:, x+jit_x, y+jit_y]
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
    F.repeat(stack.shape[0], 0)
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

    exp_x, exp_y = (stack.shape[1])//2, (stack.shape[2])//2  # these are the indices in the cross-correlation matrix that correspond to 0 shift
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


def ReAlign(stack, maxShift, co_stack=None):
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

    if co_stack is not None:
        if co_stack.shape != stack.shape:
            raise ValueError("Stack to be co-aligned needs to have same dimensions as main stack")
    total_max_shift = 0
    maxShift = int(maxShift)
    x_shifts = np.zeros(stack.shape[0])
    y_shifts = np.zeros_like(x_shifts)
    re_aligned = stack.copy()
    # compute initial sum
    sum_stack = np.sum(re_aligned, 0)
    all_t = np.arange(re_aligned.shape[0])
    np.random.shuffle(all_t)
    for t in all_t:
        xshift, yshift, sum_stack = ComputeAlignmentShift(re_aligned, t, sum_stack)
        x_shifts[t] = xshift
        y_shifts[t] = yshift
        if xshift == 0 and yshift == 0:
            continue
        if np.abs(xshift) > maxShift or np.abs(yshift) > maxShift:
            if total_max_shift < 20:
                print("Warning. Slice ", t, " requires shift greater ", maxShift, " pixels. Maximally shifted")
            elif total_max_shift == 20:
                print("More than 20 slices maximally shifted, reporting total number at end.")
            total_max_shift += 1
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
        if co_stack is not None:
            newImage = np.zeros((re_aligned.shape[1], re_aligned.shape[2]), dtype=np.float32)
            newImage[xt[0]:xt[1], yt[0]:yt[1]] = co_stack[t, xs[0]:xs[1], ys[0]:ys[1]]
            co_stack[t, :, :] = newImage
    # report back how many slices in total had to be maximally shifted
    percent_max = total_max_shift/re_aligned.shape[0]*100
    print("A total of {0} slices, or {1}% needed maximum shift".format(total_max_shift, percent_max))
    if co_stack is not None:
        return re_aligned, percent_max, x_shifts, y_shifts, co_stack
    else:
        return re_aligned, percent_max, x_shifts, y_shifts


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
        m_sum_stack = np.mean(sum_stack, 0)
        while len(centers) < nRegions * 20:
            counter += 1
            if counter > nRegions * 1000:
                break
            x = np.random.randint(radius, sum_stack.shape[1]-radius)
            y = np.random.randint(radius, sum_stack.shape[2]-radius)
            # if x < radius or y < radius or x+radius >= sum_stack.shape[1] or y+radius >= sum_stack.shape[2]:
            #     continue
            if m_sum_stack[x, y] < 2:
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


def project(u, v):
    """
    Projects the vector v orthogonally onto the line spanned by u
    """
    return np.dot(v, u) / np.dot(u, u) * u


def gram_schmidt(v, *args):
    """
    Transforms the vector v into a vector that is orthogonal to each vector
    in args and has unit length
    """
    start = v.copy()
    for u in args:
        start -= project(u, v)
    return start / np.linalg.norm(start)


def motor_nonmotor_split(activity, starting, n_reps):
    """
    For a given activity trace computes two traces that are NaN everywhere except for timepoints where bouts occured
    in some but not all repeats. For these timepoints the activity in absence and presence of the bout is returned
    Args:
        activity: A cells activity trace
        starting: The corresponding non-convovled starting trace at the same framerate
        n_reps: The number of repeats performed to identify the same timepoints with and without movement

    Returns:
        [0]: A repeat-length long trace which indicates activity of the cell in absence of motor events
        [1]: A repeat-length long trace which indicates activity of the cell in the presence of motor events
    """
    a_rep = np.reshape(activity, (n_reps, activity.size // n_reps))
    s_rep = np.reshape(starting, (n_reps, activity.size // n_reps))
    wo_motor = np.full(a_rep.shape[1], np.nan)
    w_motor = wo_motor.copy()
    if np.sum(starting) == 0:
        return wo_motor, w_motor
    motor_sums = np.sum(s_rep, 0)
    indices = np.arange(a_rep.shape[1])
    indices = indices[np.logical_and(motor_sums > 0, motor_sums < n_reps)]
    for i in indices:
        sr = s_rep[:, i]
        wo_motor[i] = np.mean(a_rep[sr == 0, i])
        w_motor[i] = np.mean(a_rep[sr > 0, i])
    return wo_motor, w_motor


def avg_motor_boost(activity, starting, n_reps):
    """
    For a given activity trace and bout starting trace tries to determine whether activity at a given repeat aligned
    timepoint is higher when there was a movement versus not
    Args:
        activity: A cells activity trace
        starting: The corresponding non-convovled starting trace at the same framerate
        n_reps: The number of repeats performed to identify the same timepoints with and without movement

    Returns:
        The average normalized change in activity for motor/vs non-motor
    """
    wo_motor, w_motor = motor_nonmotor_split(activity, starting, n_reps)
    wo = np.nansum(wo_motor)
    w = np.nansum(w_motor)
    if wo > 0:
        return w / wo
    else:
        return np.nan


def assign_region_label(centroids, region_list, res_xy, res_z=2.5):
    """
    For each centroid assigns a label of the corresponding region in region_list (note: first region matched wins)
    Args:
        centroids: The (x,y,z) centroids in um
        region_list: List of RegionContainer classes
        res_xy: Stack resolution in x/y
        res_z: Stack resolution in z

    Returns:
        Array of strings containing the labels of the region each cell belongs to or empty string if no region found
    """
    # sort region list for efficient lookup
    rlist = sorted(region_list, key=lambda rc: rc.z_index)
    rnames = []
    for cent in centroids:
        if np.any(np.isnan(cent)):
            rnames.append("")
            continue
        name = ""
        x = cent[0] / res_xy
        y = cent[1] / res_xy
        z = int(cent[2] / res_z)
        for rc in rlist:
            if rc.z_index == z:
                # NOTE: y is first row-, x second column-coordinate
                if rc.point_in_region((y, x)):
                    name = rc.region_name
                    break
            if rc.z_index > z:
                # all following regions can't match
                break
        rnames.append(name)
    return np.array(rnames)


def raster_plot(events: np.ndarray, time=None, ax=None, ticklength=1, color=None, **kwargs):
    """
    Creates a raster plot (as in spike raster plots)
    Args:
        events: An nTrials*time matrix where 0 indicates no event and values >0 indicate event times
        time: Optionally a vector to indicate timings of each column in the events matrix
        ax: The plot axis, a new plot will be created and the axis returned otherwise
        ticklength: The length of each tick
        color: Color of the ticks
        **kwargs: Additional arguments passed to vlines

    Returns:

    """
    if ax is None:
        fig, ax = pl.subplots()
    if time is None:
        time = np.arange(events.shape[1])
    for ith, trial in enumerate(events):
        trial = time[trial > 0]
        if color is None:
            ax.vlines(trial, ith + ticklength/2, ith + 1.5*ticklength, **kwargs)
        else:
            ax.vlines(trial, ith + ticklength / 2, ith + 1.5 * ticklength, colors=color, **kwargs)
    ax.set_ylim(ticklength/2, events.shape[0] + ticklength/2)
    return ax


def IndexingMatrix(trigger_frames, f_past, f_future, input_length):
    """
    Builds an indexing matrix with length(trigger_frames) rows and columns
    that index out frames from trigger_frames(n)-f_past to
    trigger_frames(n)+f_future. Indices will be clipped at [1...input_length]
    with rows containing frames outside that range removed
    Args:
        trigger_frames: Vector with all the intended trigger frames
        f_past: The number of frames before each trigger frame that should be included
        f_future: The same for the number of frames after each trigger frame
        input_length: The length of the input on which to trigger. Determines
            which rows will be removed from the matrix because they would index out
            non-existent frames
            
    Returns:
        [0]: The trigger matrix for indexing out all frames
        [1]: The number of rows that have been cut out because they would have
             contained frames with index < 0
        [2]: The number of rows that have been removed from the back because
             they would have contained frames with index >= input_length
    """
    if trigger_frames.ndim > 1:
        raise ValueError('trigger_frames has to be a vector')

    toTake = np.r_[-1 * f_past:f_future + 1]

    # turn trigger_frames into a size 2 array consisting of 1 column only
    tf = np.expand_dims(trigger_frames, 1)
    # turn toTake into a size 2 array consisting of one row only
    toTake = np.expand_dims(toTake, 0)
    # now we can use repeat to construct matrices:
    indexMat = np.repeat(tf, toTake.size, 1) + np.repeat(toTake, tf.size, 0)
    # identify front and back rows that need to be removed
    cutFront = np.sum(np.sum(indexMat < 0, 1, dtype=float) > 0, 0)
    cutBack = np.sum(np.sum(indexMat >= input_length, 1, dtype=float) > 0, 0)
    # remove out-of-bounds rows and return - if statement (seems) necessary since
    # there is no -0 for indexing the final frame
    if cutBack > 0:
        return indexMat[cutFront:-1 * cutBack, :].astype(int), cutFront, cutBack
    else:
        return indexMat[cutFront::, :].astype(int), cutFront, cutBack


def trial_average(m: np.ndarray, n_trials, sum_it=False):
    """
    Compute trial average for each trace in m
    Args:
        m: n_cells x m_timepoints matrix of activity traces
        sum_it: If true traces get summed instead of averaged
        n_trials: The number of trials

    Returns:
        n_cells x (m_timepoints // n_trials) matrix of trial averages
    """
    if m.shape[1] % n_trials != 0:
        raise ValueError("axis 1 of m has to be evenly divisible into the requested {0} trials".format(n_trials))
    if m.ndim == 2:
        m_t = np.reshape(m, (m.shape[0], n_trials, m.shape[1]//n_trials))
    elif m.ndim == 1:
        m_t = np.reshape(m, (1, n_trials, m.shape[0] // n_trials))
    else:
        raise ValueError("m has to be either 1 or 2-dimensional")
    if sum_it:
        return np.sum(m_t, 1)
    return np.mean(m_t, 1)


def trial_to_trial_correlations(mat, n_trials):
    """
    Computes trial to trial correlations of all traces in matrix
    Args:
        mat: nSamples x nTimepoints matrix of activity or behavior etc.
        n_trials: The number of trials in the dataset (mat.shape[1] % n_trials == 0)

    Returns:
        m.shape[0] long vector of average trial_to_trial correlations
    """
    def mat_mat_corr(m1: np.ndarray, m2: np.ndarray):
        """
        Computes pairwise correlations between corresponding rows in m1 and m2
        """
        ms1 = m1.copy() - np.mean(m1, 1, keepdims=True)
        ms2 = m2.copy() - np.mean(m2, 1, keepdims=True)
        norm1 = np.linalg.norm(ms1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(ms2, axis=1, keepdims=True)
        return np.sum(ms1 * ms2, axis=1, keepdims=True) / (norm1 * norm2)

    if mat.shape[1] % n_trials != 0:
        raise ValueError("Axis 1 of mat has to be evenly divisible by n_trials")
    t_len = mat.shape[1] // n_trials
    t_mats = []  # list of per-trial matrices
    for i in range(n_trials):
        t_mats.append(mat[:, i*t_len:(i+1)*t_len])
    tests = set(itertools.combinations(range(n_trials), 2))
    corrs = np.zeros(mat.shape[0])
    for t in tests:
        corrs += mat_mat_corr(t_mats[t[0]], t_mats[t[1]]).ravel()
    return corrs / len(tests)
