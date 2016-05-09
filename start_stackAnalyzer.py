import sys
from PyQt4 import QtCore, QtGui
from sta_ui import Ui_StackAnalyzer
from mh_2P import *
import numpy as np
from PIL import Image

class StartStackAnalyzer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_StackAnalyzer()
        self.ui.setupUi(self)
        self.currentStack = np.array([])
        #connect our own signals
        QtCore.QObject.connect(self.ui.btnLoad, QtCore.SIGNAL("clicked()"), self.load)
        QtCore.QObject.connect(self.ui.chkSumSlices, QtCore.SIGNAL("stateChanged(int)"), self.sumsliceschanged)

    def load(self):
        diag = QtGui.QFileDialog()
        fname = diag.getOpenFileName(self, "Select stack", "", "*.tif")
        if fname is not None and fname != "":
            assert isinstance(fname, str)
            self.currentStack = OpenStack(fname)
            self.ui.sliceView.setImage(self.currentStack)
        else:
            print("No file selected")

    def sumsliceschanged(self, state):
        if self.currentStack is None or len(self.currentStack.shape) != 3:
            print("No proper stack loaded")
            return
        if state==0:
            self.ui.sliceView.setImage(self.currentStack)
        else:
            self.ui.sliceView.setImage(np.sum(self.currentStack, 0))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = StartStackAnalyzer()
    myapp.show()
    sys.exit(app.exec_())