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
        StackAnalyzer.resize(800, 600)
        self.centralwidget = QtGui.QWidget(StackAnalyzer)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.sliceView = ImageView(self.centralwidget)
        self.sliceView.setGeometry(QtCore.QRect(10, 10, 591, 500))
        self.sliceView.setObjectName(_fromUtf8("sliceView"))
        self.btnLoad = QtGui.QPushButton(self.centralwidget)
        self.btnLoad.setGeometry(QtCore.QRect(10, 530, 75, 23))
        self.btnLoad.setObjectName(_fromUtf8("btnLoad"))
        self.chkSumSlices = QtGui.QCheckBox(self.centralwidget)
        self.chkSumSlices.setGeometry(QtCore.QRect(110, 530, 70, 17))
        self.chkSumSlices.setObjectName(_fromUtf8("chkSumSlices"))
        StackAnalyzer.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(StackAnalyzer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        StackAnalyzer.setMenuBar(self.menubar)

        self.retranslateUi(StackAnalyzer)
        QtCore.QMetaObject.connectSlotsByName(StackAnalyzer)

    def retranslateUi(self, StackAnalyzer):
        StackAnalyzer.setWindowTitle(_translate("StackAnalyzer", "2P Stack analyzer", None))
        self.btnLoad.setText(_translate("StackAnalyzer", "Load Stacks", None))
        self.chkSumSlices.setToolTip(_translate("StackAnalyzer", "Select to display slice sum", None))
        self.chkSumSlices.setText(_translate("StackAnalyzer", "Sum slices", None))

from pyqtgraph import ImageView
