# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stackanalyzer.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_StackAnalyzer(object):
    def setupUi(self, StackAnalyzer):
        StackAnalyzer.setObjectName(_fromUtf8("StackAnalyzer"))
        StackAnalyzer.resize(1286, 738)
        self.centralwidget = QtGui.QWidget(StackAnalyzer)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.sliceView = ImageView(self.centralwidget)
        self.sliceView.setGeometry(QtCore.QRect(10, 10, 641, 561))
        self.sliceView.setObjectName(_fromUtf8("sliceView"))
        self.btnLoad = QtGui.QPushButton(self.centralwidget)
        self.btnLoad.setGeometry(QtCore.QRect(10, 590, 75, 23))
        self.btnLoad.setObjectName(_fromUtf8("btnLoad"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(100, 580, 181, 131))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.rbSlices = QtGui.QRadioButton(self.groupBox)
        self.rbSlices.setGeometry(QtCore.QRect(10, 20, 121, 17))
        self.rbSlices.setObjectName(_fromUtf8("rbSlices"))
        self.rbSumProj = QtGui.QRadioButton(self.groupBox)
        self.rbSumProj.setGeometry(QtCore.QRect(10, 40, 82, 17))
        self.rbSumProj.setObjectName(_fromUtf8("rbSumProj"))
        self.rbROIOverlay = QtGui.QRadioButton(self.groupBox)
        self.rbROIOverlay.setGeometry(QtCore.QRect(10, 100, 82, 17))
        self.rbROIOverlay.setObjectName(_fromUtf8("rbROIOverlay"))
        self.rbGroupPr = QtGui.QRadioButton(self.groupBox)
        self.rbGroupPr.setGeometry(QtCore.QRect(10, 60, 71, 17))
        self.rbGroupPr.setObjectName(_fromUtf8("rbGroupPr"))
        self.spnGSize = QtGui.QSpinBox(self.groupBox)
        self.spnGSize.setGeometry(QtCore.QRect(100, 60, 71, 22))
        self.spnGSize.setMinimum(5)
        self.spnGSize.setMaximum(50)
        self.spnGSize.setProperty("value", 12)
        self.spnGSize.setObjectName(_fromUtf8("spnGSize"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(660, 10, 611, 691))
        self.tabWidget.setTabPosition(QtGui.QTabWidget.South)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.infoTab = QtGui.QWidget()
        self.infoTab.setObjectName(_fromUtf8("infoTab"))
        self.graphDFF = PlotWidget(self.infoTab)
        self.graphDFF.setGeometry(QtCore.QRect(10, 10, 581, 160))
        self.graphDFF.setObjectName(_fromUtf8("graphDFF"))
        self.graphBStarts = PlotWidget(self.infoTab)
        self.graphBStarts.setGeometry(QtCore.QRect(10, 180, 581, 160))
        self.graphBStarts.setObjectName(_fromUtf8("graphBStarts"))
        self.graphFFT = PlotWidget(self.infoTab)
        self.graphFFT.setGeometry(QtCore.QRect(10, 350, 581, 160))
        self.graphFFT.setObjectName(_fromUtf8("graphFFT"))
        self.lblFile = QtGui.QLabel(self.infoTab)
        self.lblFile.setGeometry(QtCore.QRect(10, 610, 581, 41))
        self.lblFile.setText(_fromUtf8(""))
        self.lblFile.setWordWrap(True)
        self.lblFile.setObjectName(_fromUtf8("lblFile"))
        self.tabWidget.addTab(self.infoTab, _fromUtf8(""))
        self.segTab = QtGui.QWidget()
        self.segTab.setObjectName(_fromUtf8("segTab"))
        self.groupBox_2 = QtGui.QGroupBox(self.segTab)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 581, 131))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.label_3 = QtGui.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.tbPreFrames = QtGui.QLineEdit(self.groupBox_2)
        self.tbPreFrames.setGeometry(QtCore.QRect(80, 20, 113, 20))
        self.tbPreFrames.setObjectName(_fromUtf8("tbPreFrames"))
        self.label_4 = QtGui.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(10, 50, 61, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(200, 20, 61, 16))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.tbStimFrames = QtGui.QLineEdit(self.groupBox_2)
        self.tbStimFrames.setGeometry(QtCore.QRect(80, 50, 113, 20))
        self.tbStimFrames.setObjectName(_fromUtf8("tbStimFrames"))
        self.tbPostFrames = QtGui.QLineEdit(self.groupBox_2)
        self.tbPostFrames.setGeometry(QtCore.QRect(270, 20, 113, 20))
        self.tbPostFrames.setObjectName(_fromUtf8("tbPostFrames"))
        self.tbFFTGap = QtGui.QLineEdit(self.groupBox_2)
        self.tbFFTGap.setGeometry(QtCore.QRect(270, 50, 61, 20))
        self.tbFFTGap.setObjectName(_fromUtf8("tbFFTGap"))
        self.label_6 = QtGui.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(200, 50, 61, 16))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.groupBox_3 = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(390, 10, 181, 80))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.rb6f = QtGui.QRadioButton(self.groupBox_3)
        self.rb6f.setGeometry(QtCore.QRect(10, 20, 82, 17))
        self.rb6f.setObjectName(_fromUtf8("rb6f"))
        self.rb6s = QtGui.QRadioButton(self.groupBox_3)
        self.rb6s.setGeometry(QtCore.QRect(10, 50, 82, 17))
        self.rb6s.setObjectName(_fromUtf8("rb6s"))
        self.label_9 = QtGui.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(10, 80, 101, 16))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.tbFrameRate = QtGui.QLineEdit(self.groupBox_2)
        self.tbFrameRate.setGeometry(QtCore.QRect(270, 80, 61, 20))
        self.tbFrameRate.setObjectName(_fromUtf8("tbFrameRate"))
        self.label_10 = QtGui.QLabel(self.groupBox_2)
        self.label_10.setGeometry(QtCore.QRect(200, 80, 61, 16))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.tbStimFreq = QtGui.QLineEdit(self.groupBox_2)
        self.tbStimFreq.setGeometry(QtCore.QRect(112, 80, 81, 20))
        self.tbStimFreq.setObjectName(_fromUtf8("tbStimFreq"))
        self.btnSegSave = QtGui.QPushButton(self.segTab)
        self.btnSegSave.setGeometry(QtCore.QRect(520, 630, 75, 23))
        self.btnSegSave.setObjectName(_fromUtf8("btnSegSave"))
        self.twSegment = QtGui.QTabWidget(self.segTab)
        self.twSegment.setGeometry(QtCore.QRect(10, 150, 581, 471))
        self.twSegment.setTabPosition(QtGui.QTabWidget.West)
        self.twSegment.setObjectName(_fromUtf8("twSegment"))
        self.corTab = QtGui.QWidget()
        self.corTab.setObjectName(_fromUtf8("corTab"))
        self.label_8 = QtGui.QLabel(self.corTab)
        self.label_8.setGeometry(QtCore.QRect(10, 10, 81, 16))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.tbCorrTh = QtGui.QLineEdit(self.corTab)
        self.tbCorrTh.setGeometry(QtCore.QRect(90, 10, 113, 20))
        self.tbCorrTh.setObjectName(_fromUtf8("tbCorrTh"))
        self.label_7 = QtGui.QLabel(self.corTab)
        self.label_7.setGeometry(QtCore.QRect(220, 10, 71, 16))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.tbCellDiam = QtGui.QLineEdit(self.corTab)
        self.tbCellDiam.setGeometry(QtCore.QRect(290, 10, 113, 20))
        self.tbCellDiam.setObjectName(_fromUtf8("tbCellDiam"))
        self.chkRealign = QtGui.QCheckBox(self.corTab)
        self.chkRealign.setGeometry(QtCore.QRect(420, 10, 91, 17))
        self.chkRealign.setObjectName(_fromUtf8("chkRealign"))
        self.graphAlgnSh = PlotWidget(self.corTab)
        self.graphAlgnSh.setGeometry(QtCore.QRect(10, 60, 541, 111))
        self.graphAlgnSh.setObjectName(_fromUtf8("graphAlgnSh"))
        self.twSegment.addTab(self.corTab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.twSegment.addTab(self.tab_2, _fromUtf8(""))
        self.btnSeg = QtGui.QPushButton(self.segTab)
        self.btnSeg.setGeometry(QtCore.QRect(40, 630, 75, 23))
        self.btnSeg.setObjectName(_fromUtf8("btnSeg"))
        self.tabWidget.addTab(self.segTab, _fromUtf8(""))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 630, 81, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.lcdEngines = QtGui.QLCDNumber(self.centralwidget)
        self.lcdEngines.setGeometry(QtCore.QRect(10, 650, 64, 23))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.lcdEngines.setPalette(palette)
        self.lcdEngines.setDigitCount(2)
        self.lcdEngines.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcdEngines.setObjectName(_fromUtf8("lcdEngines"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 680, 81, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        StackAnalyzer.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(StackAnalyzer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1286, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        StackAnalyzer.setMenuBar(self.menubar)

        self.retranslateUi(StackAnalyzer)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(StackAnalyzer)

    def retranslateUi(self, StackAnalyzer):
        StackAnalyzer.setWindowTitle(_translate("StackAnalyzer", "2P Stack analyzer", None))
        self.btnLoad.setText(_translate("StackAnalyzer", "Load Stack", None))
        self.groupBox.setTitle(_translate("StackAnalyzer", "Display options", None))
        self.rbSlices.setText(_translate("StackAnalyzer", "Individual slices", None))
        self.rbSumProj.setText(_translate("StackAnalyzer", "Sum slices", None))
        self.rbROIOverlay.setText(_translate("StackAnalyzer", "ROI overlay", None))
        self.rbGroupPr.setToolTip(_translate("StackAnalyzer", "Grouped projection", None))
        self.rbGroupPr.setText(_translate("StackAnalyzer", "Grouped", None))
        self.spnGSize.setToolTip(_translate("StackAnalyzer", "Group size", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.infoTab), _translate("StackAnalyzer", "Info", None))
        self.groupBox_2.setTitle(_translate("StackAnalyzer", "Experiment settings", None))
        self.label_3.setText(_translate("StackAnalyzer", "Pre Frames", None))
        self.label_4.setText(_translate("StackAnalyzer", "Stim Frames", None))
        self.label_5.setText(_translate("StackAnalyzer", "Post Frames", None))
        self.label_6.setText(_translate("StackAnalyzer", "StimFFT Gap", None))
        self.groupBox_3.setTitle(_translate("StackAnalyzer", "Indicator", None))
        self.rb6f.setText(_translate("StackAnalyzer", "Gcamp 6f", None))
        self.rb6s.setText(_translate("StackAnalyzer", "Gcamp 6s", None))
        self.label_9.setText(_translate("StackAnalyzer", "Stimulus frequency", None))
        self.label_10.setText(_translate("StackAnalyzer", "Frame rate", None))
        self.btnSegSave.setToolTip(_translate("StackAnalyzer", "Save aligned stack and graphs", None))
        self.btnSegSave.setText(_translate("StackAnalyzer", "Save", None))
        self.label_8.setText(_translate("StackAnalyzer", "Corr. threshold", None))
        self.label_7.setText(_translate("StackAnalyzer", "Cell diameter", None))
        self.chkRealign.setText(_translate("StackAnalyzer", "Realign stack", None))
        self.twSegment.setTabText(self.twSegment.indexOf(self.corTab), _translate("StackAnalyzer", "Correlation", None))
        self.twSegment.setTabText(self.twSegment.indexOf(self.tab_2), _translate("StackAnalyzer", "Tab 2", None))
        self.btnSeg.setText(_translate("StackAnalyzer", "Segment", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.segTab), _translate("StackAnalyzer", "Segmentation", None))
        self.label.setText(_translate("StackAnalyzer", "Parallel pool", None))
        self.label_2.setText(_translate("StackAnalyzer", "engines running", None))

from pyqtgraph import ImageView, PlotWidget
