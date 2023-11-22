# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 23:05:40 2017

@author: miran
"""
import sys
import math
import traceback

import numpy as np

from PyQt5.QtWidgets import QWidget, QLabel, QApplication, \
                            QPushButton, QHBoxLayout, QFrame, \
                            QVBoxLayout, QProgressDialog, \
                            QMessageBox, QColorDialog, QSlider
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtCore import Qt, QCoreApplication, QThread
from PyQt5.QtCore import pyqtSignal as QSignal

from widgets import resources, stylesheet, dialogs


def prepareqt(icon=None, style_sheet=None, newapp=False):
    qapp = QCoreApplication.instance()

    if icon is None:
        icon = 'lambda.png'

    if style_sheet is None:
        style_sheet = stylesheet.defaultStyleSheet()

    if qapp is None or newapp:
        qapp = QApplication(sys.argv)
        qapp.setWindowIcon(resources.loadIcon(icon))
        qapp.setStyleSheet(style_sheet)

        resources.loadResources() # cache the resources
    return qapp

def centerOnScreen(widget):
    rect = QApplication.desktop().screenGeometry()
    x = (rect.width() - widget.width())//2
    y = (rect.height() - widget.height())//2
    widget.move(x, y)

def horizontalLine():
    hline = QFrame()
    hline.setFrameShape(QFrame.HLine)
    hline.setFrameShadow(QFrame.Sunken)
    return hline

def verticalLine():
    vline = QFrame()
    vline.setFrameShape(QFrame.VLine)
    vline.setFrameShadow(QFrame.Sunken)
    return vline


class QShowHidePushButton(QPushButton):
    def __init__(self, state=False, parent=None):
        QPushButton.__init__(self, parent)
        self.setCheckable(True)
        self._toggled(state)
        self.toggled.connect(self._toggled)
        self.setChecked(state)

    def _toggled(self, state):
        self.setIcon(
            resources.loadIcon(
                {False: 'hide.png', True: 'show.png'}.get(state, 'hide.png')
            )
        )


class QDrawErasePushButton(QPushButton):
    def __init__(self, state=False, parent=None):
        QPushButton.__init__(self, parent)
        self.setCheckable(True)
        self._toggled(state)
        self.toggled.connect(self._toggled)
        self.setChecked(state)

    def _toggled(self, state):
        self.setIcon(
            resources.loadIcon(
                {False: 'erase.png', True: 'draw.png'}.get(state, 'erase.png')
            )
        )

class QLockUnlockPushButton(QPushButton):
    def __init__(self, state=False, parent=None):
        QPushButton.__init__(self, parent)
        self.setCheckable(True)
        self._toggled(state)
        self.toggled.connect(self._toggled)
        self.setChecked(state)

    def _toggled(self, state):
        self.setIcon(
            resources.loadIcon(
                {False: 'unlocked.png', True: 'locked.png'}.get(state, 'unlocked.png')
            )
        )


class QCheckablePushButton(QPushButton):
    def __init__(self, state=False, parent=None):
        QPushButton.__init__(self, parent)
        self.setCheckable(True)
        self._toggled(state)
        self.toggled.connect(self._toggled)
        self.setChecked(state)

    def _toggled(self, state):
        self.setIcon(
            resources.loadIcon(
                {False: 'cancel.png', True: 'ok.png'}.get(state, 'cancel.png')
            )
        )


class QThinPushButton(QPushButton):
    def __init__(self, icon=None, parent=None):
        QPushButton.__init__(self, parent)
        if icon is not None:
            if isinstance(icon, str):
                icon = resources.loadIcon(icon)
            self.setIcon(icon)


class QSquarePushButton(QPushButton):
    def __init__(self, icon=None, parent=None):
        QPushButton.__init__(self, parent)
        if icon is not None:
            if isinstance(icon, str):
                icon = resources.loadIcon(icon)
            self.setIcon(icon)


class QColorSelectionButton(QPushButton):
    colorChanged = QSignal(object)
    colorChangedEx = QSignal(object, object)

    def __init__(self, color, *args, **kwargs):
        QPushButton.__init__(self, *args, **kwargs)
        self._color = None
        self.setColor(color)
        self.clicked.connect(self.getColor)

    def setColor(self, color):
        color = QColor(color)
        if color != self._color:
            style = 'QColorSelectionButton {{background-color: {}}};'.format(
                color.name())
            self.setStyleSheet(style)
            old_color = self._color
            self._color = color
            self.colorChanged.emit(self._color)
            self.colorChangedEx.emit(self._color, old_color)

    def getColor(self):
        if self._color is not None:
            color = QColorDialog.getColor(self._color, self)
            if color.isValid():
                self.setColor(color)

    def color(self):
        return self._color


#!! This section will be moved to module dialogs.py
def ErrorMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Critical, title, label,
                      QMessageBox.Ok, parent)
    dlg.button(QMessageBox.Ok).setIcon(resources.loadIcon('ok.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def ErrorQuestion(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Critical, title, label,
                      QMessageBox.Yes | QMessageBox.No, parent)
    dlg.button(QMessageBox.Yes).setIcon(resources.loadIcon('ok.png'))
    dlg.button(QMessageBox.No).setIcon(resources.loadIcon('cancel.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def WarningMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Warning, title, label,
                      QMessageBox.Ok, parent)
    dlg.button(QMessageBox.Ok).setIcon(resources.loadIcon('ok.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def InformationMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Information, title, label,
                      QMessageBox.Ok, parent)
    dlg.button(QMessageBox.Ok).setIcon(resources.loadIcon('ok.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def QuestionMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Question, title, label,
                      QMessageBox.Yes | QMessageBox.No, parent)
    dlg.button(QMessageBox.Yes).setIcon(resources.loadIcon('ok.png'))
    dlg.button(QMessageBox.No).setIcon(resources.loadIcon('cancel.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def WarningQuestion(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Warning, title, label,
                      QMessageBox.Yes | QMessageBox.No, parent)
    dlg.button(QMessageBox.Yes).setIcon(resources.loadIcon('ok.png'))
    dlg.button(QMessageBox.No).setIcon(resources.loadIcon('cancel.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg
#!! This section will be moved to module dialogs.py


class QClickableLabel(QLabel):
    doubleClicked = QSignal()

    def __init(self, parent):
        QLabel.__init__(self, parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self.eventFilter)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space:
            self.doubleClicked.emit()


class QAdjustWidget(QWidget):
    valueChanged = QSignal(float)

    class Transform:
        def __init__(self):
            self._adjust = None
        def toSlider(self, userValue):
            return userValue
        def fromSlider(self, sliderValue):
            return sliderValue
        def str(self, sliderValue):
            return str(sliderValue)
        def setParent(self, adjust):
            self._adjust = adjust

    class FpTransform:
        def __init__(self, minimum, maximum, step, ndecimals=None):
            self._step = float(step)

            minimum = float(minimum)
            self._minimum = math.ceil(minimum/self._step)*self._step

            maximum = float(maximum)
            self._maximum = math.floor(maximum/self._step)*self._step

            self._delta = float(maximum - minimum)

            self._page = float(step*10)
            self._n = int(self._delta/self._step)

            if ndecimals is None:
                ndecimals = int(max(math.ceil(math.log10(1.0/self._step)), 1))
            self._fmtstr = '{{:.{}f}}'.format(int(ndecimals))

            self._adjust = None

        def setParent(self, adjust):

            self._adjust = adjust

            self._adjust.slider.setMinimum(0)
            self._adjust.slider.setMaximum(self._n)

            self._adjust.slider.setSingleStep(1)
            self._adjust.slider.setPageStep(10)

        def toSlider(self, value):
            return int(round((value - self._minimum)/self._step))

        def fromSlider(self, position):
            value = position*self._step + self._minimum
            return value

        def str(self, position):
            value = self.fromSlider(position)
            return self._fmtstr.format(value)

    class FpLogTransform:
        def __init__(self, minimum, maximum, n=100, ndecimals=None):
            self._n = int(n)

            minimum = float(minimum)
            self._log_minimum = math.log(minimum)

            maximum = float(maximum)
            self._log_maximum = math.log(maximum)

            self._log_delta = self._log_maximum - self._log_minimum

            self._n = int(n)

            if ndecimals is None:
                ndecimals = max(math.ceil(abs(math.log10(minimum))), 1)
            self._fmtstr = '{{:.{}f}}'.format(int(ndecimals))

            self._adjust = None

        def setParent(self, adjust):

            self._adjust = adjust

            self._adjust.slider.setMinimum(0)
            self._adjust.slider.setMaximum(self._n)

            self._adjust.slider.setSingleStep(1)
            self._adjust.slider.setPageStep(10)

        def toSlider(self, value):
            pos = int(round((math.log(value) - self._log_minimum)/
                            self._log_delta*self._n))
            return pos

        def fromSlider(self, position):
            value = math.exp(
                position*self._log_delta/self._n + self._log_minimum)
            return value

        def str(self, position):
            value = self.fromSlider(position)
            return self._fmtstr.format(value)

    NO_TRANSFORM = Transform()

    def __init__(self, parent=None, captionloc='top'):
        QWidget.__init__(self, parent)

        self._transform = QAdjustWidget.NO_TRANSFORM

        captionLayout = QHBoxLayout()
        sliderLayout = QHBoxLayout()

        self.captionLabel = QLabel()
        self.valueLabel = QLabel()
        self.unitsLabel = QLabel()
        self.previewPositionLabel = QLabel()
        self.previewPositionLabel.setVisible(False)
        self._setValueLabel(0)

        self.gotoStartButton = QThinPushButton('gotostart.png')
        self.gotoStartButton.setToolTip(
            QCoreApplication.translate('QAdjustWidget', 'To minimum'))
        self.gotoStartButton.clicked.connect(
            lambda: self.slider.setValue(self.slider.minimum()))
        self.gotoStartButton.setVisible(False)

        self.decrementButton = QSquarePushButton('decrement.png')
        self.decrementButton.setAutoRepeat(True)
        self.decrementButton.setToolTip(
            QCoreApplication.translate('QAdjustWidget', 'Decrement'))
        self.decrementButton.clicked.connect(
            lambda: self.slider.setValue(self.slider.value() - 1))

        self.incrementButton = QSquarePushButton('increment.png')
        self.incrementButton.setAutoRepeat(True)
        self.incrementButton.setToolTip(
            QCoreApplication.translate('QAdjustWidget', 'Increment'))
        self.incrementButton.clicked.connect(
            lambda: self.slider.setValue(self.slider.value() + 1))

        self.gotoEndButton = QThinPushButton('gotoend.png')
        self.gotoEndButton.setToolTip(
            QCoreApplication.translate('QAdjustWidget', 'To maximum'))
        self.gotoEndButton.clicked.connect(
            lambda: self.slider.setValue(self.slider.maximum()))
        self.gotoEndButton.setVisible(False)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._setValueLabel)
        self.slider.valueChanged.connect(
            lambda value: self.valueChanged.emit(
                self._transform.fromSlider(value))
        )
        self.slider.setToolTip(
            QCoreApplication.translate('QAdjustWidget', 'Adjust'))
        self.slider.setTracking(True)
        self.slider.sliderMoved.connect(self._positionTracking)
        self.slider.sliderPressed.connect(self._startPositionTracking)
        self.slider.sliderReleased.connect(
            lambda: self.previewPositionLabel.setVisible(False))

        if captionloc in ('top', 'bottom'):
            captionLayout.addStretch()
            captionLayout.addWidget(self.captionLabel)
            captionLayout.addWidget(self.valueLabel)
            captionLayout.addWidget(self.previewPositionLabel)
            captionLayout.addWidget(self.unitsLabel)
            captionLayout.addStretch()
        elif captionloc == 'left':
            captionLayout.addStretch()
            captionLayout.addWidget(self.captionLabel)
            captionLayout.addWidget(self.valueLabel)
            captionLayout.addWidget(self.previewPositionLabel)
            captionLayout.addWidget(self.unitsLabel)
        elif captionloc == 'right':
            captionLayout.addWidget(self.captionLabel)
            captionLayout.addWidget(self.valueLabel)
            captionLayout.addWidget(self.previewPositionLabel)
            captionLayout.addWidget(self.unitsLabel)
            captionLayout.addStretch()
        else:
            raise ValueError("Parameter captionloc had an illegal value! "\
                             "Use one of ('top', 'bottom', 'left', 'right')")

        sliderLayout.addWidget(self.gotoStartButton)
        sliderLayout.addWidget(self.decrementButton)
        sliderLayout.addWidget(self.slider)
        sliderLayout.addWidget(self.incrementButton)
        sliderLayout.addWidget(self.gotoEndButton)

        if captionloc == 'top':
            mainLayout = QVBoxLayout()
            mainLayout.setContentsMargins(0, 0, 0, 0)
            mainLayout.addLayout(captionLayout)
            mainLayout.addLayout(sliderLayout)
        elif captionloc == 'bottom':
            mainLayout = QVBoxLayout()
            mainLayout.setContentsMargins(0, 0, 0, 0)
            mainLayout.addLayout(sliderLayout)
            mainLayout.addLayout(captionLayout)
        elif captionloc == 'left':
            mainLayout = QHBoxLayout()
            mainLayout.setContentsMargins(0, 0, 0, 0)
            mainLayout.addLayout(captionLayout)
            mainLayout.addLayout(sliderLayout)
        elif captionloc == 'right':
            mainLayout = QHBoxLayout()
            mainLayout.setContentsMargins(0, 0, 0, 0)
            mainLayout.addLayout(sliderLayout)
            mainLayout.addLayout(captionLayout)

        #mainLayout.addStretch(1)

        self.setLayout(mainLayout)

    def setCaptionVisible(self, state):
        self.captionLabel.setVisible(state)
        self.valueLabel.setVisible(state)
        self.previewPositionLabel.setVisible(state)
        self.unitsLabel.setVisible(state)

    def setTransform(self, transform):
        if transform is None:
            transform = QAdjustWidget.NO_TRANSFORM

        if transform != self._transform:
            transform.setParent(self)
            self._transform = transform
            self._setValueLabel(self.slider.value())

    def transform(self):
        return self._transform

    def _setValueLabel(self, value):
        self.valueLabel.setText(self._transform.str(value))

    def _startPositionTracking(self):
        if not self.slider.hasTracking():
            self.previewPositionLabel.setText(
                '(' + self._transform.str(self.slider.value()) + ')')
            self.previewPositionLabel.setVisible(True)

    def _positionTracking(self, position):
        if not self.slider.hasTracking():
            self.previewPositionLabel.setText(
                '(' + self._transform.str(position) + ')')

    def setIncrementToolTip(self, tooltip):
        self.incrementButton.setToolTip(tooltip)
    def incrementToolTip(self):
        return self.incrementButton.toolTip()

    def setTracking(self, state):
        self.slider.setTracking(state)
    def tracking(self):
        return self.slider.hasTracking()

    def setDecrementToolTip(self, tooltip):
        self.decrementButton.setToolTip(tooltip)
    def decrementToolTip(self):
        return self.decrementButton.toolTip()

    def setSliderToolTip(self, tooltip):
        self.slider.setToolTip(tooltip)
    def sliderToolTip(self):
        return self.slider.toolTip()

    def setGotoStartToolTip(self, tooltip):
        self.gotoStartButton.setToolTip(tooltip)
    def gotoStartToolTip(self):
        return self.gotoStartButton.toolTip()

    def setGotoEndToolTip(self, tooltip):
        self.gotoEndButton.setToolTip(tooltip)
    def gotoEndToolTip(self):
        return self.gotoEndButton.toolTip()

    def setGotoStartVisible(self, state):
        self.gotoStartButton.setVisible(state)
    def gotoStartVisible(self):
        return self.gotoStartButton.visible()

    def setGotoEndVisible(self, state):
        self.gotoEndButton.setVisible(state)
    def gotoEndVisible(self):
        return self.gotoEndButton.visible()

    def setGotoEndsVisible(self, state):
        self.gotoEndButton.setVisible(state)
        self.gotoStartButton.setVisible(state)

    def setMinimum(self, minimum):
        self.slider.setMinimum(self._transform.toSlider(minimum))
    def minimum(self):
        return self._transform.fromSlider(self.slider.minimum())

    def setMaximum(self, maximum):
        self.slider.setMaximum(self._transform.toSlider(maximum))
    def maximum(self):
        return self._transform.fromSlider(self.slider.maximum())

    def setRange(self, minimum, maximum):
        self.slider.setRange(self._transform.toSlider(minimum),
                             self._transform.toSlider(maximum))
    def range(self):
        return (self._transform.fromSlider(self.slider.minimum()),
                self._transform.fromSlider(self.slider.maximum()))

    def setStep(self, step):
        self.slider.setSingleStep(self._transform.toSlider(step))
    def step(self):
        return self._transform.fromSlider(self.slider.singleStep())

    def setPage(self, page):
        self.slider.setPageStep(self._transform.toSlider(page))
    def page(self):
        return self._transform.fromSlider(self.slider.pageStep())

    def setValue(self, value):
        value = self._transform.toSlider(value)
        if value != self.slider.value():
            self.slider.setValue(value)
    def value(self):
        return self._transform.fromSlider(self.slider.value())

    def setUnits(self, units):
        self.unitsLabel.setText(units)
    def units(self):
        return self.unitsLabel.text()

    def setCaption(self, text):
        self.captionLabel.setText(text)
    def capton(self):
        return self.captionLabel.text()

    def setEnabled(self, state):
        state = not state
        self.gotoStartButton.setDisabled(state)
        self.decrementButton.setDisabled(state)
        self.slider.setDisabled(state)
        self.incrementButton.setDisabled(state)
        self.gotoEndButton.setDisabled(state)
        self.setDisabled(state)

    def enabled(self):
        return self.isEnabled()

class QBackgroundTask(QThread):
    statusChanged = QSignal(str)
    progressChanged = QSignal(int)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self._parent = parent
        self._canceled = None
        self._errDetailedStr = self._errStr = None

        self._target = self._args = self._kwargs = None
        self._title = self._result = self._dlg = None
        self._canceled = False

    # GUI thread side API
    def start(self, target=None, args=(), kwargs=None,
              title=None, label=None, button=None):

        if kwargs is None:
            kwargs = {}

        if button is None:
            button = self.tr('Cancel')

        self._target = target
        self._args = args
        self._kwargs = kwargs

        self._title = title
        self._result = self._errDetailedStr = self._errStr = None
        self._canceled = False

        self._dlg = QProgressDialog(label, button, 0, 100, self._parent)
        self._dlg.setWindowModality(Qt.WindowModal)
        if title is not None:
            self._dlg.setWindowTitle(title)
        self._dlg.setWindowFlags(self._dlg.windowFlags() &
                                 ~Qt.WindowContextHelpButtonHint)
        self._dlg.canceled.connect(self.cancel)
        self.progressChanged.connect(self._dlg.setValue)
        self.statusChanged.connect(self._dlg.setLabelText)
        self.finished.connect(self._finalize)

        QThread.start(self)
        self._dlg.exec_()
        self.wait()

        return not self._canceled and self._errStr is None

    def stop(self):
        self._dlg.cancel()

    def result(self):
        return self._result

    def _finalize(self):
        self._dlg.hide()
        self._dlg.reset()

        if self._errStr is not None or self._errDetailedStr is not None:
            errDlg = QMessageBox(
                QMessageBox.Critical,
                self._title,
                self._errStr,
                QMessageBox.Ok,
                self._parent
            )
            if self._errDetailedStr is not None:
                errDlg.setDetailedText(self._errDetailedStr)
            errDlg.exec_()

    def error(self):
        if self._errStr is not None or self._errDetailedStr is not None:
            return self._errStr, self._errDetailedStr
        return None

    # Worker thread side API
    def setProgress(self, percent):
        self.progressChanged.emit(int(percent))

    def setStatus(self, status):
        self.statusChanged.emit(str(status))

    def wasCanceled(self):
        return self._canceled

    def reportError(self, description, details=None):
        self._errStr = description
        self._errDetailedStr = details

    def cancel(self):
        self._canceled = True

    def run(self):
        result = None
        try:
            self._result = self._target(*self._args, **self._kwargs)
        except:
            self.reportError(
                QCoreApplication.translate(
                    'QBackgroundTask',
                    'Unexpected error in the worker thread!'),
                traceback.format_exc()
            )
        return result

class QBackgroundTaskEx(QThread):
    statusChanged = QSignal(str)
    progressChanged = QSignal(int)
    subtaskComplete = QSignal(str, int)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self._parent = parent
        self._canceled = None
        self._errDetailedStr = self._errStr = None

        self._target = self._args = self._kwargs = None
        self._title = self._result = self._dlg = None
        self._canceled = False

    # GUI thread side API
    def start(self, target=None, args=(), kwargs=None,
              title=None, label=None, button=None, showdetails=False):

        if kwargs is None:
            kwargs = {}

        if button is None:
            button = self.tr('Cancel')

        self._target = target
        self._args = args
        self._kwargs = kwargs

        self._title = title
        self._result = self._errDetailedStr = self._errStr = None
        self._canceled = False

        self._dlg = dialogs.QTaskProgressDialog(
            label=label, title=title, parent=self._parent)
        self._dlg.setShowDetails(showdetails)
        self._dlg.setWindowModality(Qt.WindowModal)
        self._dlg.setWindowFlags(self._dlg.windowFlags() &
                                 ~Qt.WindowContextHelpButtonHint)
        self._dlg.canceled.connect(self.cancel)
        self.progressChanged.connect(self._dlg.setValue)
        self.statusChanged.connect(self._dlg.setLabelText)
        self.finished.connect(self._finalize)
        self.subtaskComplete.connect(self._dlg.appendItem)

        QThread.start(self)
        self._dlg.exec_()
        self.wait()

        return not self._canceled and self._errStr is None

    def stop(self):
        self._dlg.cancel()

    def result(self):
        return self._result

    def _finalize(self):
        self._dlg.hide()
        self._dlg.reset()

        if self._errStr is not None or self._errDetailedStr is not None:
            errDlg = QMessageBox(
                QMessageBox.Critical,
                self._title,
                self._errStr,
                QMessageBox.Ok,
                self._parent
            )
            if self._errDetailedStr is not None:
                errDlg.setDetailedText(self._errDetailedStr)
            errDlg.exec_()

    def error(self):
        if self._errStr is not None or self._errDetailedStr is not None:
            return self._errStr, self._errDetailedStr
        return None

    # Worker thread side API
    def setProgress(self, percent):
        self.progressChanged.emit(int(percent))

    def setStatus(self, status):
        self.statusChanged.emit(str(status))

    def reportSubtask(self, description, status):
        self.subtaskComplete.emit(str(description), int(status))

    def wasCanceled(self):
        return self._canceled

    def reportError(self, description, details=None):
        self._errStr = description
        self._errDetailedStr = details

    def cancel(self):
        self._canceled = True

    def run(self):
        result = None
        try:
            self._result = self._target(*self._args, **self._kwargs)
        except:
            self.reportError(
                QCoreApplication.translate(
                    'QBackgroundTask',
                    'Unexpected error in the worker thread!'),
                traceback.format_exc()
            )
        return result

def GsNp2QImage(npimg, depth=None):
    siz = npimg.shape
    # rawdepth = npimg.itemsize*8
    if npimg.dtype == np.uint8:
        if depth is not None and depth != 8:
            npimg = np.array(npimg << (8 - depth), dtype=np.uint8)
    elif npimg.dtype == np.uint16:
        if depth is not None and depth < 16:
            npimg = np.array(npimg >> (depth - 8), dtype=np.uint8)
    elif npimg.dtype == np.uint32:
        if depth is not None and depth < 32:
            npimg = np.array(npimg << (depth - 8), dtype=np.uint8)
    elif npimg.dtype == np.float32:
        k = 255.0/npimg.max()
        if depth is not None:
            k = 255.0/(2**depth - 1.0)
        else:
            k = 255.0/npimg.max()
        npimg = np.array(npimg*k + 0.5, dtype=np.uint8).tobytes()
    elif npimg.dtype == np.float64:
        k = 255.0/npimg.max()
        if depth is not None:
            k = 255.0/(2**depth - 1.0)
        else:
            k = 255.0/npimg.max()
        npimg = np.array(npimg*k + 0.5, dtype=np.uint8).tobytes() # TODO: find a more elaborate solution

    return QImage(npimg, siz[1], siz[0], QImage.Format_Indexed8), npimg
