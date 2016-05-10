import sys
from PyQt4 import QtCore, QtGui
from sta_ui import Ui_StackAnalyzer
from mh_2P import *
import numpy as np
import pickle
from PIL import Image

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
            self.ui.rbSumProj.setCheckable(True)
            self.ui.rbSumProj.setEnabled(True)

        else:
            print("No file selected")

    def displayChanged(self, check):
        if self.currentStack is None or len(self.currentStack.shape) != 3:
            print("No proper stack loaded")
            return
        if self.ui.rbSlices.isChecked():
            self.ui.sliceView.setImage(self.currentStack)
        elif self.ui.rbSumProj.isChecked():
            self.ui.sliceView.setImage(np.sum(self.currentStack, 0))
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
                pi = self.ui.graphDFF.plotItem
                pi.clear()
                pi.plot(self.percentileDff(g.RawTimeseries), pen=(255, 0, 0))
                pi.showGrid(x=True, y=True)
                pi.setLabel("left", "dF/F0")
                pi.setLabel("bottom", "Frames")
                pi.setTitle("Raw dF/F0")





if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = StartStackAnalyzer()
    myapp.show()
    sys.exit(app.exec_())
