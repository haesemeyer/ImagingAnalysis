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
        StackAnalyzer.resize(1024, 677)
        self.centralwidget = QtGui.QWidget(StackAnalyzer)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.sliceView = ImageView(self.centralwidget)
        self.sliceView.setGeometry(QtCore.QRect(10, 10, 591, 500))
        self.sliceView.setObjectName(_fromUtf8("sliceView"))
        self.btnLoad = QtGui.QPushButton(self.centralwidget)
        self.btnLoad.setGeometry(QtCore.QRect(10, 530, 75, 23))
        self.btnLoad.setObjectName(_fromUtf8("btnLoad"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(100, 520, 181, 131))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.rbSlices = QtGui.QRadioButton(self.groupBox)
        self.rbSlices.setGeometry(QtCore.QRect(10, 20, 121, 17))
        self.rbSlices.setObjectName(_fromUtf8("rbSlices"))
        self.rbSumProj = QtGui.QRadioButton(self.groupBox)
        self.rbSumProj.setGeometry(QtCore.QRect(10, 40, 82, 17))
        self.rbSumProj.setObjectName(_fromUtf8("rbSumProj"))
        self.rbROIOverlay = QtGui.QRadioButton(self.groupBox)
        self.rbROIOverlay.setGeometry(QtCore.QRect(10, 60, 82, 17))
        self.rbROIOverlay.setObjectName(_fromUtf8("rbROIOverlay"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(600, 10, 421, 501))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.infoTab = QtGui.QWidget()
        self.infoTab.setObjectName(_fromUtf8("infoTab"))
        self.graphDFF = PlotWidget(self.infoTab)
        self.graphDFF.setGeometry(QtCore.QRect(10, 10, 401, 131))
        self.graphDFF.setObjectName(_fromUtf8("graphDFF"))
        self.graphBStarts = PlotWidget(self.infoTab)
        self.graphBStarts.setGeometry(QtCore.QRect(10, 150, 401, 131))
        self.graphBStarts.setObjectName(_fromUtf8("graphBStarts"))
        self.graphFFT = PlotWidget(self.infoTab)
        self.graphFFT.setGeometry(QtCore.QRect(10, 290, 401, 131))
        self.graphFFT.setObjectName(_fromUtf8("graphFFT"))
        self.lblFile = QtGui.QLabel(self.infoTab)
        self.lblFile.setGeometry(QtCore.QRect(10, 430, 401, 41))
        self.lblFile.setText(_fromUtf8(""))
        self.lblFile.setWordWrap(True)
        self.lblFile.setObjectName(_fromUtf8("lblFile"))
        self.tabWidget.addTab(self.infoTab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        StackAnalyzer.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(StackAnalyzer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 21))
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
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.infoTab), _translate("StackAnalyzer", "Info", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("StackAnalyzer", "Tab 2", None))

from pyqtgraph import ImageView, PlotWidget
