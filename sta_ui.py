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
        self.corrTab = QtGui.QWidget()
        self.corrTab.setObjectName(_fromUtf8("corrTab"))
        self.tabWidget.addTab(self.corrTab, _fromUtf8(""))
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
        self.tabWidget.setCurrentIndex(0)
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
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.corrTab), _translate("StackAnalyzer", "Corr. segmentation", None))
        self.tabWidget.setTabToolTip(self.tabWidget.indexOf(self.corrTab), _translate("StackAnalyzer", "Correlation based analysis", None))
        self.label.setText(_translate("StackAnalyzer", "Parallel pool", None))
        self.label_2.setText(_translate("StackAnalyzer", "engines running", None))

from pyqtgraph import ImageView, PlotWidget
