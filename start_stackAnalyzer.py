import sys
from PyQt4 import QtCore, QtGui
from sta_ui import Ui_StackAnalyzer
from mh_2P import *
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import warnings

class StartStackAnalyzer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_StackAnalyzer()
        self.ui.setupUi(self)
        self.currentStack = np.array([])
        self.graphList = []
        # set ui defaults
        self.ui.rbSumProj.setCheckable(False)
        self.ui.rbROIOverlay.setCheckable(False)
        self.ui.rbSumProj.setEnabled(False)
        self.ui.rbROIOverlay.setEnabled(False)
        self.ui.rbSlices.setChecked(True)
        # hide menu and roi button on image view
        self.ui.sliceView.ui.roiBtn.hide()
        self.ui.sliceView.ui.menuBtn.hide()
        # connect our own signals
        QtCore.QObject.connect(self.ui.btnLoad, QtCore.SIGNAL("clicked()"), self.load)
        QtCore.QObject.connect(self.ui.rbROIOverlay, QtCore.SIGNAL("toggled(bool)"), self.displayChanged)
        QtCore.QObject.connect(self.ui.rbSumProj, QtCore.SIGNAL("toggled(bool)"), self.displayChanged)
        QtCore.QObject.connect(self.ui.rbSlices, QtCore.SIGNAL("toggled(bool)"), self.displayChanged)
        # react to click on pixel
        self.ui.sliceView.getImageItem().mouseClickEvent = self.sliceViewClick

    # Helper functions #
    @staticmethod
    def getAlignedName(tiffname):
        return tiffname[:-4]+"_stack.npy"

    @staticmethod
    def getGraphName(tiffname):
        return tiffname[:-3]+"graph"

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

    def clearGraphs(self):
        self.ui.graphDFF.plotItem.clear()
        self.ui.graphBStarts.plotItem.clear()
        self.ui.graphFFT.plotItem.clear()

    def plotGraphInfo(self, graph):
        self.clearGraphs()
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
        ts = np.sum(self.currentStack, 2)
        ts = np.sum(ts, 1)
        pi.plot(self.percentileDff(ts), pen=(150, 50, 50))
        pi.showGrid(x=True, y=True)
        pi.setLabel("left", "dF/F0")
        pi.setLabel("bottom", "Frames")
        pi.setTitle("Slice summed dF/F0")


    # Signals #

    def load(self):
        diag = QtGui.QFileDialog()
        fname = diag.getOpenFileName(self, "Select stack", "E:/Dropbox/2P_Data", "*.tif")
        if fname is not None and fname != "":
            assert isinstance(fname, str)
            self.graphList = []
            try:
                self.currentStack = np.load(self.getAlignedName(fname)).astype(float)
                print("Loaded pre-aligned stack")
            except FileNotFoundError:
                self.currentStack = OpenStack(fname)
            try:
                f = open(self.getGraphName(fname), 'rb')
                self.graphList = pickle.load(f)
                self.ui.rbROIOverlay.setCheckable(True)
                self.ui.rbROIOverlay.setEnabled(True)
                print("Loaded graph file")
            except FileNotFoundError:
                self.ui.rbROIOverlay.setCheckable(False)
                self.ui.rbROIOverlay.setEnabled(False)
            self.ui.sliceView.setImage(self.currentStack)
            self.ui.lblFile.setText(fname)
            # reset ui options and clear unit graphs
            self.ui.rbSumProj.setCheckable(True)
            self.ui.rbSumProj.setEnabled(True)
            self.ui.rbSlices.setChecked(True)
            self.clearGraphs()
            self.plotStackAverageFluorescence()

        else:
            print("No file selected")

    def displayChanged(self, check):
        if self.currentStack is None or len(self.currentStack.shape) != 3:
            print("No proper stack loaded")
            return
        if self.ui.rbSlices.isChecked():
            self.ui.sliceView.setImage(self.currentStack)
            self.plotStackAverageFluorescence()
        elif self.ui.rbSumProj.isChecked():
            self.ui.sliceView.setImage(np.sum(self.currentStack, 0))
            self.plotStackAverageFluorescence()
        elif self.ui.rbROIOverlay.isChecked():
            proj = self.getROIProjection()
            if proj is None:
                print("No ROIs in graph-list")
                return
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





if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = StartStackAnalyzer()
    myapp.show()
    sys.exit(app.exec_())
