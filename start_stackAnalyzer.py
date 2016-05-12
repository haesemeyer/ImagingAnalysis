from PyQt4 import QtCore, QtGui
from sta_ui import Ui_StackAnalyzer
from mh_2P import *
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import warnings
from time import perf_counter
try:
    from ipyparallel import Client
except NameError:
    from IPython.parallel import Client


class StartStackAnalyzer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_StackAnalyzer()
        self.ui.setupUi(self)
        self.currentStack = np.array([])
        self.graphList = []
        self.filename = ""
        # set ui defaults
        self.ui.rbSumProj.setCheckable(False)
        self.ui.rbROIOverlay.setCheckable(False)
        self.ui.rbSumProj.setEnabled(False)
        self.ui.rbGroupPr.setEnabled(False)
        self.ui.rbROIOverlay.setEnabled(False)
        self.ui.rbGroupPr.setEnabled(False)
        self.ui.rbSlices.setChecked(True)
        self.ui.tbPreFrames.setText("72")
        self.ui.tbStimFrames.setText("144")
        self.ui.tbPostFrames.setText("180")
        self.ui.tbFFTGap.setText("48")
        self.ui.tbMinPhot.setText("0.02")
        self.ui.rb6s.setChecked(True)
        self.ui.tbCorrTh.setText("0.5")
        self.ui.tbCellDiam.setText("8")
        self.ui.tbStimFreq.setText("0.1")
        self.ui.tbFrameRate.setText("2.4")
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.btnSeg.setEnabled(False)
        self.ui.btnSegSave.setEnabled(False)
        # hide menu and roi button on image views
        self.ui.sliceView.ui.roiBtn.hide()
        self.ui.sliceView.ui.menuBtn.hide()
        self.ui.segColView.ui.roiBtn.hide()
        self.ui.segColView.ui.menuBtn.hide()
        # connect our own signals
        QtCore.QObject.connect(self.ui.btnLoad, QtCore.SIGNAL("clicked()"), self.load)
        QtCore.QObject.connect(self.ui.rbROIOverlay, QtCore.SIGNAL("toggled(bool)"), self.displayChanged)
        QtCore.QObject.connect(self.ui.rbSumProj, QtCore.SIGNAL("toggled(bool)"), self.displayChanged)
        QtCore.QObject.connect(self.ui.rbSlices, QtCore.SIGNAL("toggled(bool)"), self.displayChanged)
        QtCore.QObject.connect(self.ui.btnSeg, QtCore.SIGNAL("clicked()"), self.segment)
        QtCore.QObject.connect(self.ui.btnSegSave, QtCore.SIGNAL("clicked()"), self.saveSegmentation)
        # react to click on pixel
        self.ui.sliceView.getImageItem().mouseClickEvent = self.sliceViewClick
        # create our cash dictionary
        self._cash = dict()
        # try connecting to ipython pool
        try:
            self._rc = Client(timeout=0.5)
        except OSError:
            self._rc = []
            print("No parallel pool found")
        self.ui.lcdEngines.display(str(len(self._rc)))

    # UI element property access #
    @property
    def PreFrames(self):
        return int(self.ui.tbPreFrames.text())

    @property
    def StimFrames(self):
        return int(self.ui.tbStimFrames.text())

    @property
    def PostFrames(self):
        return int(self.ui.tbPostFrames.text())

    @property
    def FFTGap(self):
        return int(self.ui.tbFFTGap.text())

    @property
    def StimulusFrequency(self):
        return float(self.ui.tbStimFreq.text())

    @property
    def FrameRate(self):
        return float(self.ui.tbFrameRate.text())

    @property
    def MinPhot(self):
        return float(self.ui.tbMinPhot.text())

    @property
    def CaTimeConstant(self):
        if self.ui.rb6f.isChecked():
            return 400 / 1000
        elif self.ui.rb6s.isChecked():
            return 1796 / 1000
        else:
            raise ValueError("Unknown indicator")

    @property
    def CorrThresh(self):
        return float(self.ui.tbCorrTh.text())

    @property
    def CellDiam(self):
        return int(self.ui.tbCellDiam.text())

    # Helper functions #
    @staticmethod
    def getAlignedName(tiffname):
        return tiffname[:-4]+"_stack.npy"

    @staticmethod
    def getGraphName(tiffname):
        return tiffname[:-3]+"graph"

    @staticmethod
    def getTailName(tiffname):
        return tiffname[:-6]+".tail"

    def findROIByPixel(self, x, y):
        if len(self.graphList) == 0:
            return
        for g in self.graphList:
            if g.MinX <= x <= g.MaxX and g.MinY <= y <= g.MaxY:
                # possible overlap
                for v in g.V:
                    if v[0] == x and v[1] == y:
                        return g
        return None

    def getROIProjection(self):
        if len(self.graphList) == 0:
            return None
        proj = np.zeros((self.currentStack.shape[1], self.currentStack.shape[2], 3))
        s = np.sum(self.currentStack, 0)
        s /= (np.max(s) * 1.2)
        proj[:, :, 0] = proj[:, :, 1] = proj[:, :, 2] = s
        for g in self.graphList:
            colChoice = np.random.randint(0, 6)
            for v in g.V:
                if colChoice == 0 or colChoice == 3 or colChoice == 4:
                    proj[v[0], v[1], 0] = 1
                if colChoice == 1 or colChoice == 3 or colChoice == 5:
                    proj[v[0], v[1], 1] = 1
                if colChoice == 2 or colChoice == 4 or colChoice == 5:
                    proj[v[0], v[1], 2] = 1
        return proj

    @staticmethod
    def percentileDff(timeseries, percentile=20):
        f0 = np.percentile(timeseries, percentile)
        if f0 == 0:
            print("F0 was 0")
            f0 = 1
        return (timeseries - f0) / f0

    @staticmethod
    def computeAveragedTrace(graph, overhang=1):
        """
        Uses information in the graph class to try and determine how many per-plane
        repetitions where performed within the experiment and then computes
        a repeat-averaged version of the timeseries.
        Args:
            graph: Unit graph
            overhang: Number of 'extra frames' recorded

        Returns:
            A repetition averaged version of the graph's raw timeseries
        """
        l = graph.RawTimeseries.size
        perRep = graph.FramesPre + graph.FramesStim + graph.FramesPost
        nRep = (l - overhang) / perRep
        return np.sum(np.reshape(graph.RawTimeseries[0:l-overhang], (nRep, (l-overhang)//nRep)), 0)

    @staticmethod
    def computeStimFourier(graph, aggregate=True):

        def ZScore(t):
            return (t - np.mean(t)) / np.std(t)

        # anti-aliasing
        a = StartStackAnalyzer.computeAveragedTrace(graph)
        filtered = gaussian_filter1d(a, graph.FrameRate / 8)
        sf = graph.FramesPre + graph.FramesFFTGap
        ef = graph.FramesPre + graph.FramesStim
        filtered = filtered[sf:ef]
        # TODO: Somehow make the following noise reduction more generally applicable...
        # if the length of filtered is divisble by 2, break into two blocks and average for noise reduction
        if aggregate:
            # Test if we can aggregate: Find the period length pl in frames. If the length of filtered
            # is a multiple of 2 period lengths (size = 2* N * pl), reshape and average across first
            # and second half to reduce noise in transform (while at the same time reducing resolution)
            plen = round(1 / graph.StimFrequency * graph.FrameRate)
            if (filtered.size / plen) % 2 == 0:
                filtered = np.mean(filtered.reshape((2, filtered.size // 2)), 0)
            else:
                warnings.warn('Could not aggregate for fourier due to phase alignment mismatch')
        fft = np.fft.rfft(ZScore(filtered))
        freqs = np.linspace(0, graph.FrameRate / 2, fft.shape[0])
        ix = np.argmin(np.absolute(graph.StimFrequency - freqs))  # index of bin which contains our desired frequency
        return freqs, fft, ix

    def groupedProjection(self, groupSize):
        nSlices = self.currentStack.shape[0] // groupSize
        grouped = np.zeros((nSlices, self.currentStack.shape[1], self.currentStack.shape[2]))
        for i in range(nSlices):
            grouped[i, :, :] = np.sum(self.currentStack[i*groupSize:(i+1)*groupSize, :, :], 0)
        return grouped

    def sliceCorrelations(self, filterWin=(5, 1, 1)):
        # NOTE: No performance is gained by parallelizing this operation. Too much overhead, corrcoef already parallel
        fstack = gaussian_filter(self.currentStack, filterWin)
        st = np.sum(self.currentStack, 0).flatten()
        slc = np.zeros(self.currentStack.shape[0])
        for i in range(self.currentStack.shape[0]):
            slc[i] = np.corrcoef(st, fstack[i, :, :].flatten())[0, 1]
        return slc

    def assignMotorToGraphs(self):
        """
        Tries to load taildata corresponding to graph-list elements
        under the assumption that they all belong to the same
        experiment. Assigns corresponding data to each graph
        """
        if self.graphList is None or len(self.graphList) == 0:
            return
        g = self.graphList[0]
        td = TailData.LoadTailData(self.getTailName(g.SourceFile), g.CaTimeConstant, 100)
        if td is None:
            print("No tail tracking file found")
            return
        pfv = td.PerFrameVigor
        bst = td.FrameBoutStarts(g.FrameRate)
        for g in self.graphList:
            g.PerFrameVigor = pfv
            g.BoutStartTrace = bst

    def chunkWork(self, a, axis=0, nchunks=None):
        """
        Takes the array a and divides it into chunks for each parallel
        pool engine along indicated axis
        Args:
            a: The array to divide
            axis: The axis of division

        Returns: List of chunks for each engine or None if no engines exist

        """
        if len(self._rc) == 0:
            return None
        if nchunks is None:
            poolsize = len(self._rc)
        else:
            poolsize = nchunks
        return np.array_split(a, poolsize, axis=axis)


    def clearPlots(self):
        self.ui.graphDFF.plotItem.clear()
        self.ui.graphBStarts.plotItem.clear()
        self.ui.graphFFT.plotItem.clear()

    def plotGraphInfo(self, graph):
        self.clearPlots()
        # plot dff
        pi = self.ui.graphDFF.plotItem
        pi.plot(self.percentileDff(graph.RawTimeseries), pen=(255, 0, 0))
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "dF/F0")
        pi.setLabel("bottom", "Frames")
        pi.setTitle("Raw dF/F0")
        # plot bout starts
        pi = self.ui.graphBStarts.plotItem
        if graph.BoutStartTrace is not None:
            pi.plot(graph.BoutStartTrace, pen=(0, 0, 255))
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "Bout starts")
        pi.setLabel("bottom", "Frames")
        pi.setTitle("Convolved bout start trace")
        freqs, fft, ix = self.computeStimFourier(graph)
        mags = np.absolute(fft)
        pi = self.ui.graphFFT.plotItem
        pi.plot(freqs, mags, pen=(255, 0, 0))
        pi.plot([freqs[ix], freqs[ix]], [0, mags.max()])
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "Magnitude")
        pi.setLabel("bottom", "Frequency", "Hz")
        pi.setTitle("Stimulus fourier transform")

    def plotStackAverageFluorescence(self):
        pi = self.ui.graphDFF.plotItem
        if "stack_ts" in self._cash:
            ts = self._cash["stack_ts"]
        else:
            ts = np.sum(self.currentStack, 2)
            ts = np.sum(ts, 1)
            self._cash["stack_ts"] = ts
        pi.plot(ts, pen=(150, 50, 50))
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "Photon count")
        pi.setLabel("bottom", "Frames")
        pi.setTitle("Slice summed photon count")
        pi = self.ui.graphBStarts.plotItem
        pi.plot(self.percentileDff(ts), pen=(150, 50, 50))
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "Slice dF/F0")
        pi.setLabel("bottom", "Frames")
        pi.setTitle("Slice average dF/F0")
        pi = self.ui.graphFFT.plotItem
        if "stack_slcorr" in self._cash:
            sliceCorr = self._cash["stack_slcorr"]
        else:
            sliceCorr = self.sliceCorrelations()
            self._cash["stack_slcorr"] = sliceCorr
        pi.plot(sliceCorr, pen=(50, 150, 50))
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "Correlation")
        pi.setLabel("bottom", "Frames")
        pi.setTitle("Slice correlation to mean stack")

    def segmentByCorrelation(self):
        if self.ui.chkRealign.checkState():
            maxshift = int(self.ui.tbCellDiam.text()) // 2
            self.currentStack, xshift, yshift = ReAlign(self.currentStack, maxshift)
            # remove borders that may have been affected by the realignment
            self.currentStack[:, :maxshift, :] = 0
            self.currentStack[:, :, :maxshift] = 0
            self.currentStack[:, self.currentStack.shape[1] - maxshift:, :] = 0
            self.currentStack[:, :, self.currentStack.shape[2] - maxshift] = 0
            pi = self.ui.graphAlgnSh.plotItem
            pi.clear()
            pi.plot(xshift, pen=(150, 0, 0))
            pi.plot(yshift, pen=(0, 150, 50))
            pi.showGrid(x=True, y=True)
            pi.setLabel("left", "Shift [px]")
            pi.setLabel("bottom", "Frames")
            pi.setTitle("Slice shifts in x and y direction")
        filterWins = (self.FrameRate, self.CellDiam // 8, self.CellDiam // 8)
        if len(self._rc) > 0:
            # perform shuffled correlation computation on a different engine
            ar_sh = self._rc[0].apply_async(ComputeShuffledCorrelations, self.currentStack, filterWins, self.MinPhot)
            im_nc_shuff = []  # place holder definition
        else:
            im_nc_shuff = ComputeShuffledCorrelations(self.currentStack, filterWins,
                                                      self.MinPhot*self.currentStack.shape[0])
        # compute neighborhood correlations on original stack
        sum_stack = np.sum(self.currentStack, 0)
        consider = lambda x, y: sum_stack[x, y] >= (self.MinPhot*self.currentStack.shape[0])
        rate_stack = gaussian_filter(self.currentStack, filterWins)
        im_ncorr = AvgNeighbhorCorrelations(rate_stack, 2, consider)
        print('Maximum neighbor correlation in stack = ', im_ncorr.max(), flush=True)
        seed_cutoff = 1
        if len(self._rc) > 0:
            # need to pick up our shuffled stack!
            im_nc_shuff = ar_sh.get()
        for c in np.linspace(0, 1, 1001):
            if ((im_ncorr > c).sum() / (im_nc_shuff > c).sum()) >= 10:
                seed_cutoff = c
                break
        print('Correlation seed cutoff in stack = ', seed_cutoff, flush=True)
        # extract correlation graphs - 4-connected
        # cap our growth correlation threshold at the seed-cutoff, i.e. if corr_thresh
        # is larger than the significance threshold reduce it, when creating graph
        if self.CorrThresh <= seed_cutoff:
            ct_actual = self.CorrThresh
        else:
            ct_actual = seed_cutoff
        graph, colors = CorrelationGraph.CorrelationConnComps(rate_stack, im_ncorr, ct_actual, consider, False,
                                                              (0, rate_stack.shape[0]), seed_cutoff)
        self.ui.segColView.setImage(colors)
        min_size = np.pi * (self.CellDiam / 2) ** 2 / 2  # half of a circle with the given average cell diameter
        graph = [g for g in graph if g.NPixels >= min_size]  # remove compoments with less than 30 pixels
        print('Identified ', len(graph), 'units in slice ', flush=True)
        # assign necessary information to each graph
        for g in graph:
            g.SourceFile = self.filename  # store for convenience access
            g.FramesPre = self.PreFrames
            g.FramesStim = self.StimFrames
            g.FramesPost = self.PostFrames
            g.FramesFFTGap = self.FFTGap
            g.StimFrequency = self.StimulusFrequency
            g.CellDiam = self.CellDiam
            g.CorrThresh = ct_actual
            g.CorrSeedCutoff = seed_cutoff
            g.RawTimeseries = np.zeros_like(g.Timeseries)
            g.FrameRate = self.FrameRate
            g.CaTimeConstant = self.CaTimeConstant
            # TODO: Find replacement for quality score deviation or re-compute!
            for v in g.V:
                g.RawTimeseries = g.RawTimeseries + self.currentStack[:, v[0], v[1]]
        return graph

    # Signals #

    def load(self):
        diag = QtGui.QFileDialog()
        fname = diag.getOpenFileName(self, "Select stack", "E:/Dropbox/2P_Data", "*.tif")
        if fname is not None and fname != "":
            assert isinstance(fname, str)
            self.graphList = []
            try:
                self.currentStack = np.load(self.getAlignedName(fname)).astype(float)
                # if we found an aligned stack, set the corresponding segmentation option to false
                self.ui.chkRealign.setChecked(False)
                print("Loaded pre-aligned stack")
            except FileNotFoundError:
                self.currentStack = OpenStack(fname)
                self.ui.chkRealign.setChecked(True)
            try:
                f = open(self.getGraphName(fname), 'rb')
                self.graphList = pickle.load(f)
                self.ui.rbROIOverlay.setCheckable(True)
                self.ui.rbROIOverlay.setEnabled(True)
                print("Loaded graph file")
            except FileNotFoundError:
                self.ui.rbROIOverlay.setCheckable(False)
                self.ui.rbROIOverlay.setEnabled(False)
            self.ui.lblFile.setText(fname)
            self.resetAfterLoad()
            self.filename = fname
        else:
            print("No file selected")

    def displayChanged(self, check):
        if self.currentStack is None or len(self.currentStack.shape) != 3:
            print("No proper stack loaded")
            return
        if self.ui.rbSlices.isChecked():
            self.clearPlots()
            self.ui.sliceView.setImage(self.currentStack)
            self.plotStackAverageFluorescence()
        elif self.ui.rbSumProj.isChecked():
            self.clearPlots()
            if "sumStack" in self._cash:
                self.ui.sliceView.setImage(self._cash["sumStack"])
            else:
                sumStack = np.sum(self.currentStack, 0)
                self.ui.sliceView.setImage(sumStack)
                self._cash["sumStack"] = sumStack
            self.plotStackAverageFluorescence()
        elif self.ui.rbROIOverlay.isChecked():
            if "roiProjection" in self._cash:
                proj = self._cash["roiProjection"]
            else:
                proj = self.getROIProjection()
                self._cash["roiProjection"] = proj
            if proj is None:
                print("No ROIs in graph-list")
                return
            self.ui.sliceView.setImage(proj)
        elif self.ui.rbGroupPr.isChecked():
            if ("grProj", self.ui.spnGSize.value()) in self._cash:
                proj = self._cash[("grProj", self.ui.spnGSize.value())]
            else:
                proj = self.groupedProjection(self.ui.spnGSize.value())
                self._cash[("grProj", self.ui.spnGSize.value())] = proj
            self.ui.sliceView.setImage(proj)

        else:
            print("Unknown display option")

    def sliceViewClick(self, event):
        if not self.ui.rbROIOverlay.isChecked():
            return
        event.accept()
        pos = event.pos()
        x, y = int(pos.x()), int(pos.y())
        print(x, y)
        print(np.sum(self.currentStack[:, x, y]))
        if self.ui.rbROIOverlay.isChecked():
            g = self.findROIByPixel(x, y)
            if g is not None:
                print("Graph size = ", g.NPixels)
                self.plotGraphInfo(g)

    def segment(self):
        if self.ui.twSegment.currentIndex() == 0:
            # correlation based segmentation
            self.graphList = self.segmentByCorrelation()
            self.ui.rbROIOverlay.setCheckable(True)
            self.ui.rbROIOverlay.setEnabled(True)
            self.resetAfterLoad()
        else:
            print("Not implemented segmentation option selected")
        # if possible assign motor information
        self.assignMotorToGraphs()

    def saveSegmentation(self):
        pass

    def resetAfterLoad(self):
        self.ui.sliceView.setImage(self.currentStack)
        # clear our cash
        self._cash.clear()
        # reset ui options and clear unit graphs
        self.ui.rbSumProj.setCheckable(True)
        self.ui.rbSumProj.setEnabled(True)
        self.ui.rbGroupPr.setEnabled(True)
        self.ui.rbGroupPr.setCheckable(True)
        self.ui.rbSlices.setChecked(True)
        self.ui.btnSeg.setEnabled(True)
        self.clearPlots()
        self.plotStackAverageFluorescence()


# parallel pool helpers
def ComputeShuffledCorrelations(stack, filterdims, minphot):
    from mh_2P import ShuffleStackTemporal, AvgNeighbhorCorrelations
    from scipy.ndimage import gaussian_filter
    import numpy as np
    st_shuff = ShuffleStackTemporal(stack)
    sum_stack = np.sum(stack, 0)
    consider = lambda x, y: sum_stack[x, y] >= minphot
    rs_shuff = gaussian_filter(st_shuff, filterdims)
    return AvgNeighbhorCorrelations(rs_shuff, 2, consider)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = StartStackAnalyzer()
    myapp.show()
    sys.exit(app.exec_())
