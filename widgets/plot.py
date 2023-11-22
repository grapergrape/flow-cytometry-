# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:49:03 2017

@author: miran
"""
import sys
import time
import os.path
import pickle
import traceback

import numpy as np
import scipy.io

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
                            QLabel, QCheckBox, \
                            QSplitter, QFileDialog, QSizePolicy
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtCore import pyqtSignal as QSignal

import pyqtgraph as pg
from pyqtgraph import exporters

# exporting mkPen
from pyqtgraph import mkPen

from widgets import common, dialogs

PICKLE_PROTOCOL = 3

class StatusBar(QWidget):
    sigFrozen = QSignal(bool)
    export = QSignal()

    def __init__(self, plot=None):
        QWidget.__init__(self)

        self._refPos = None
        self._hLine = self._vLine = None
        self._refHLine = self._refVLine = None

        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        self._infoLabel = QLabel()
        self._positionLabel = QLabel('x=\ny=')
        self._cursorLineCheckBox = QCheckBox()
        self._cursorLineCheckBox.setText(
            QCoreApplication.translate('StatusBar', 'Measure'))
        self._cursorLineCheckBox.setTristate(False)
        self._cursorLineCheckBox.setToolTip(
            QCoreApplication.translate('StatusBar', 'Show/Hide cursor lines.'))
        self._cursorLineCheckBox.stateChanged.connect(
            self.setCursorLineVisible)
        self._cursorLineColorButton = common.QColorSelectionButton(
            QColor('#00aaff'))
        self._cursorLineColorButton.colorChanged.connect(
            self._setCursorLineColor)
        self._cursorLineColorButton.setToolTip(
            QCoreApplication.translate('StatusBar',
                                       'Change cursor line color.'))
        self._refCursorLineColorButton = common.QColorSelectionButton(Qt.gray)
        self._refCursorLineColorButton.colorChanged.connect(
            self._setRefCursorLineColor)
        self._refCursorLineColorButton.setToolTip(
            QCoreApplication.translate('StatusBar',
                                       'Change reference cursor line color.'))

        self.setPlot(plot)

        self._measurementLabel = QLabel('dx=\ndy=')

        self._gridCheckbox = QCheckBox()
        self._gridCheckbox.setText(
            QCoreApplication.translate('StatusBar', 'Grid'))
        self._gridCheckbox.setTristate(False)
        self._gridCheckbox.setToolTip(
            QCoreApplication.translate('StatusBar', 'Turn on/off the grid.'))
        self._gridCheckbox.stateChanged.connect(
            self.setGridVisible)

        self._logxCheckbox = QCheckBox()
        self._logxCheckbox.setText(
            QCoreApplication.translate('StatusBar', 'Log x'))
        self._logxCheckbox.setTristate(False)
        self._logxCheckbox.setToolTip(
            QCoreApplication.translate('StatusBar', 'Log scale for x axis.'))
        self._logxCheckbox.stateChanged.connect(
            self.setLogXScale)

        self._logyCheckbox = QCheckBox()
        self._logyCheckbox.setText(
            QCoreApplication.translate('StatusBar', 'Log y'))
        self._logyCheckbox.setTristate(False)
        self._logyCheckbox.setToolTip(
            QCoreApplication.translate('StatusBar', 'Log scale for y axis.'))
        self._logyCheckbox.stateChanged.connect(
            self.setLogYScale)

        self._frozenCheckBox = common.QLockUnlockPushButton()
        self._frozenCheckBox.toggled.connect(self._frozenChanged)
        self._frozenCheckBox.setToolTip(
            QCoreApplication.translate(
                'ImageStatusBar',
                'Enable/disable updates.'))

        self._exportPushButton = common.QSquarePushButton('export.png')
        self._exportPushButton.clicked.connect(lambda: self.export.emit())
        self._exportPushButton.setToolTip(
            QCoreApplication.translate(
                'ImageStatusBar',
                'Export data to a file.'))

        layout.addWidget(self._infoLabel)
        layout.addWidget(self._positionLabel)
        layout.addStretch()
        layout.addWidget(self._measurementLabel)
        layout.addStretch()
        logLayout = QVBoxLayout()
        logLayout.setContentsMargins(0, 0, 0, 0)
        logLayout.addWidget(self._logxCheckbox)
        logLayout.addWidget(self._logyCheckbox)
        layout.addLayout(logLayout)
        gridMeasureLayout = QVBoxLayout()
        gridMeasureLayout.setContentsMargins(0, 0, 0, 0)
        gridMeasureLayout.addWidget(self._cursorLineCheckBox)
        gridMeasureLayout.addWidget(self._gridCheckbox)
        layout.addLayout(gridMeasureLayout)
        layout.addWidget(self._cursorLineColorButton)
        layout.addWidget(self._refCursorLineColorButton)
        layout.addWidget(self._frozenCheckBox)
        layout.addWidget(self._exportPushButton)

        self.setCursorLineVisible(True)
        self.setLayout(layout)
        self.sizePolicy().setVerticalPolicy(QSizePolicy.Minimum)

    def _frozenChanged(self, value):
        self.sigFrozen.emit(value)

    def setFrozen(self, value):
        self._frozenCheckBox.setChecked(value)

    def frozen(self):
        return bool(self._frozenCheckBox.isChecked())

    def _setCursorLineColor(self, color):
        pen = pg.mkPen(color=color, style=Qt.DotLine)
        if self._hLine is not None:
            self._hLine.setPen(pen)
        if self._vLine is not None:
            self._vLine.setPen(pen)

    def _setRefCursorLineColor(self, color):
        pen = pg.mkPen(color=color, style=Qt.DotLine)
        if self._refHLine is not None:
            self._refHLine.setPen(pen)
        if self._refVLine is not None:
            self._refVLine.setPen(pen)

    def setPlotInfo(self, info):
        if info is None:
            self._infoLabel.clear()
        else:
            self._infoLabel.setText(info)

    def setPosition(self, pos):
        self._positionLabel.setText('x={:g}\ny={:g}'.format(
            pos.x(), pos.y()))

    def setCursorLineVisible(self, state):
        self._cursorLineCheckBox.setChecked(state)
        #if state and self._plot:
        #    self._plot.autoRange()

        if self._vLine is not None:
            self._vLine.setVisible(state)
        if self._hLine is not None:
            self._hLine.setVisible(state)
        self._cursorLineCheckBox.setChecked(state)

        self._refHLine.setVisible(state and self._refPos is not None)
        self._refVLine.setVisible(state and self._refPos is not None)
        if not state:
            self._measurementLabel.clear()

    def cursorLineVisible(self):
        return self._cursorLineCheckBox.isChecked()

    def setGridVisible(self, state):
        self._gridCheckbox.setChecked(state)
        self._plot.plotWidget().showGrid(x=state, y=state)

    def setLogXScale(self, state):
        self._logxCheckbox.setChecked(state)
        self._plot.plotWidget().logScale(x=state)

    def setLogYScale(self, state):
        self._logyCheckbox.setChecked(state)
        self._plot.plotWidget().logScale(y=state)

    def setPlot(self, plot):
        self._plot = plot
        plot.scene().sigMouseMoved.connect(self._mouseMove)
        plot.scene().sigMouseClicked.connect(self._mouseClicked)

        pen = pg.mkPen(
            color=self._cursorLineColorButton.color(), style=Qt.DotLine)
        self._vLine = plot.vLine(0, pen=pen)
        self._hLine = plot.hLine(0, pen=pen)

        pen = pg.mkPen(
            color=self._refCursorLineColorButton.color(), style=Qt.DotLine)
        self._refHLine = plot.hLine(0, pen=pen)
        self._refVLine = plot.vLine(0, pen=pen)
        self._refHLine.setVisible(False)
        self._refVLine.setVisible(False)

    def _mouseMove(self, pos):
        try:
            plotpos = self._plot.plotItem.vb.mapSceneToView(pos)
            #plotpos = self._plot.mapFromScene(pos)
            self.setPosition(plotpos)
            self._vLine.setValue(plotpos.x())
            self._hLine.setValue(plotpos.y())
            self._setMeasurement(plotpos)
        except:
            print(sys.exc_info())

    def _setMeasurement(self, pos):
        if self._refPos is not None and self.cursorLineVisible():
            dx = pos.x() - self._refPos.x()
            dy = pos.y() - self._refPos.y()
            self._measurementLabel.setText(
                'dx={:g}\ndy={:g}'.format(dx, dy))
        else:
            self._measurementLabel.clear()

    def _mouseClicked(self, event):
        try:
            plotpos = self._plot.plotItem.vb.mapSceneToView(event.scenePos())

            if not event.isAccepted() and not event.double():
                self._refHLine.setValue(plotpos.y())
                self._refVLine.setValue(plotpos.x())
                self._refPos = plotpos

                cursorLineVisible = self.cursorLineVisible()
                self._refHLine.setVisible(cursorLineVisible)
                self._refVLine.setVisible(cursorLineVisible)
            #plotpos = self._plot.mapFromScene(pos)
            self.setPosition(plotpos)
            self._vLine.setValue(plotpos.x())
            self._hLine.setValue(plotpos.y())
        except:
            print(sys.exc_info())

class Plot(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        #!! clear = self.clear
        pg.PlotWidget.__init__(self, *args, **kwargs)

        self._defaultXRange = self._defaultYRange = None

        #!! self.clear = clear
        self.getPlotItem().setMenuEnabled(False, False)
        self._items = {}
        #self.mouseDoubleClickEvent = lambda event: self.autoRange()
        self._vlines = {}
        self._hlines = {}

    def mouseDoubleClickEvent(self, event):
        if Qt.ControlModifier & event.modifiers():
            if self._defaultXRange is not None:
                self.setXRange(*self._defaultXRange)
            if self._defaultYRange is not None:
                self.setYRange(*self._defaultYRange)
        else:
            self.autoRange()

        super().mouseDoubleClickEvent(event)

    def plot(self, *args, **kwargs):
        item = self.getPlotItem().plot(*args, **kwargs)
        if 'plotid' in kwargs:
            self._items[kwargs['plotid']] = item
        return item

    def getPlot(self, plotid):
        return self._items.get(plotid)

    def removePlot(self, plotid):
        if plotid in self._items:
            self.removeItem(self._items.pop(plotid))

    def removePlots(self):
        for item in self._items:
            self.removeItem(item)
        self._items = {}

    def setData(self, plotid, *args, **kwargs):
        if plotid not in self._items:
            self.plot(*args, plotid=plotid, **kwargs)
        else:
            self._items[plotid].setData(*args, **kwargs)

    def getData(self, plotid):
        return self._items[plotid].getData()

    def item(self, itemid):
        return self._items.get(itemid)

    def items(self):
        return self._items

    def setXLabel(self, label, *args, **kwargs):
        self.getPlotItem().setLabel('bottom', label, *args, **kwargs)

    def setYLabel(self, label, *args, **kwargs):
        self.getPlotItem().setLabel('left', label, *args, **kwargs)

    def setTitle(self, title, *args, **kwargs):
        self.getPlotItem().setTitle(title, *args, **kwargs)

    def vLine(self, pos, lineid=None, cb=None, **kwargs):
        lineitem = self.getPlotItem().addLine(x=pos, **kwargs)
        if lineid is not None:
            self._vlines[lineid] = lineitem
        if cb is not None:
            lineitem.sigPositionChanged.connect(cb)
        return lineitem

    def setVLinePosition(self, lineid, pos, *args, **kwargs):
        if lineid in self._vlines:
            self._vlines[lineid].setValue(pos)
        else:
            self._vlines[lineid] = self.vLine(pos, *args, **kwargs)
    def getVLinePosition(self, lineid):
        if lineid in self._vlines:
            return self._vlines[lineid].getXPos()

    def getVLine(self, lineid):
        return self._vlines.get(lineid)

    def removeVline(self, lineid):
        if lineid in self._vlines:
            vline = self._vlines.pop(lineid)
            self.plotItem.removeItem(vline)

    def removeVLines(self):
        for key in self._vlines:
            self.plotItem.removeItem(self._vlines[key])
        self._vlines.clear()

    def hLine(self, pos, lineid=None, cb=None, **kwargs):
        lineitem = self.getPlotItem().addLine(y=pos, **kwargs)
        if lineid is not None:
            self._hlines[lineid] = lineitem
        if cb is not None:
            lineitem.sigPositionChanged.connect(cb)
        return lineitem

    def setHLinePosition(self, lineid, pos, *args, **kwargs):
        if lineid in self._hlines:
            self._hlines[lineid].setValue(pos)
        else:
            self._hlines[lineid] = self.hLine(pos, *args, **kwargs)
    def getHLinePosition(self, lineid):
        if lineid in self._hlines:
            return self._hlines[lineid].getYPos()

    def getHLine(self, lineid):
        return self._hlines.get(lineid)

    def removeHline(self, lineid):
        if lineid in self._hlines:
            hline = self._hlines.pop(lineid)
            self.plotItem.removeItem(hline)

    def removeHLines(self):
        for key in self._hlines:
            self.plotItem.removeItem(self._hlines[key])
        self._hlines.clear()

    def clear(self):
        self.removeHLines()
        self.removeVLines()

        self.removePlots()
        if self.getPlotItem().legend is not None:
            self.getPlotItem().legend.close()
            self.getPlotItem().legend = None
        self.getPlotItem().clear()

    def grid(self, x=None, y=None, opacity=1.0):
        self.getPlotItem().showGrid(x, y, opacity)

    def logScale(self, x=None, y=None):
        xrange = None
        if x is None:
            xrange = self.getAxis('bottom').range
        self.getPlotItem().setLogMode(x, y)
        self.disableAutoRange()
        if xrange is not None:
            self.setXRange(*xrange, padding=0)

    def setDefaultXRange(self, low, high):
        self._defaultXRange = (float(low), float(high))

    def setDefaultYRange(self, low, high):
        self._defaultYRange = (float(low), float(high))

class Image(QWidget):
    def __init__(self, **kwargs):

        QWidget.__init__(self, **kwargs)
        self._hsplitter = QSplitter(Qt.Horizontal)
        self._image = QWidget()
        self._histogram = QWidget()

        self._hsplitter.addWidget(self._image)
        self._hsplitter.addWidget(self._histogram)

        self._mainlayout = QHBoxLayout()
        self._mainlayout.setContentsMargins(0, 0, 0, 0)
        self._mainlayout.setSpacing(0)
        self._mainlayout.addWidget(self._hsplitter)

        self.setLayout(self._mainlayout)

class Plotx(QWidget):
    CHECKBOX_START_INDEX = 5

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self._plot = Plot(self)
        self._statusBar = StatusBar(self)
        self._statusBar.setMaximumHeight(
            self._statusBar.minimumSizeHint().height())
        self._statusBar.sigFrozen.connect(self.setFrozen)

        self._splitter = QSplitter()
        self._splitter.setOrientation(Qt.Vertical)
        self._splitter.addWidget(self._plot)
        self._splitter.addWidget(self._statusBar)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        self._statusBar.export.connect(self.export)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        self.setLayout(layout)

    def plotWidget(self):
        return self._plot

    def setPlotInfo(self, info):
        self._statusBar.setPlotInfo(info)

    def statusBar(self):
        return self._statusBar

    def setShowStatusBar(self, state):
        if state:
            self.showStatusBar()
        else: self.hideStatusBar()
    def hideStatusBar(self):
        self._splitter.setSizes([1, 0])
        #self._statusBar.setVisible(False)
    def showStatusBar(self):
        self._splitter.setSizes([1, 1])
        #self._statusBar.setVisible(True)

    def setFrozen(self, state):
        self._statusBar.setFrozen(bool(state))
    def frozen(self):
        return self._statusBar.frozen()

    def setGridVisible(self, state):
        self._statusBar.setGridVisible(state)
    def showGrid(self):
        self._statusBar.setGridVisible(True)
    def hideGrid(self):
        self._statusBar.setGridVisible(False)

    def setCursorLineVisible(self, state):
        self._statusBar.setCursorLineVisible(state)
    def showCursorLine(self):
        self._statusBar.setCursorLineVisible(True)
    def hideCursorLine(self):
        self._statusBar.setCursorLineVisible(False)
    def logScale(self, x=None, y=None):
        if x is not None:
            self._statusBar.setLogXScale(x)
        if y is not None:
            self._statusBar.setLogYScale(y)
        self._plot.logScale(x, y)

    def exportData(self):
        counter = 0
        data = {}
        plotItemsDict = self.items()
        for item_id, item in plotItemsDict.items():
            x = y = None
            try:
                x = item.xData
                y = item.yData
            except AttributeError:
                continue

            if x is not None or y is not None:
                data['x_{:d}'.format(counter)] = x
                data['y_{:d}'.format(counter)] = y
                counter += 1
        return data

    def export(self, filename=None, usematplotlib=False):
        if filename is None:
            filename = QFileDialog.getSaveFileName(
                self,
                QCoreApplication.translate('Plot', 'Export to file'),
                '',
                QCoreApplication.translate(
                    'Plot',
                    'PNG image (*.png);;'\
                    'TIF image (*.tif);;'\
                    'JPG image (*.jpg);;'\
                    'SVG vector graphics (*.svg);;'\
                    'NPZ compressed numpy data (*.npz);;'\
                    'PKL Python pickle (*.pkl);;'\
                    'MAT Matlab data file (*.mat)'
                )
            )
            if isinstance(filename, tuple):
                filename = filename[0]

        if filename:
            root, ext = os.path.splitext(filename)
            ext = ext.lower()

            plotItem = self.plotWidget().getPlotItem()

            try:
                if usematplotlib and ext in ('.png', '.tif', '.jpg'):
                    exporter = exporters.MatplotlibExporter(plotItem)
                    exporter.export(filename)
                else:
                    if ext in ('.png', '.tif', '.jpg'):
                        exporter = exporters.ImageExporter(plotItem)
                        exporter.export(filename)

                    elif ext == '.svg':
                        exporter = exporters.SVGExporter(plotItem)
                        exporter.export(filename)

                    elif ext in ('.npz', '.pkl', '.mat'):
                        data = self.exportData()
                        if data:
                            if ext == '.npz':
                                np.savez_compressed(filename, **data)
                            elif ext == '.pkl':
                                with open(filename, 'wb') as fid:
                                    pickle.dump(data, fid, PICKLE_PROTOCOL)
                            elif ext == '.mat':
                                scipy.io.savemat(
                                    filename, data, do_compression=True)
                        else:
                            dialogs.InformationMessage(
                                self,
                                QCoreApplication.translate(
                                    'Plotx', 'Export data'),
                                QCoreApplication.translate(
                                    'Plotx', 'Nothing to export')
                            )
            except:
                dlg = dialogs.ErrorMessage(
                    self,
                    QCoreApplication.translate('Plotx', 'Export data'),
                    QCoreApplication.translate(
                        'Plotx',
                        'Falied to export data to the specified file!'
                    ),
                    traceback.format_exc()
                )
                dlg.exec_()

    def __getattr__(self, name):
        return getattr(self._plot, name)

class LivePlotx(Plotx):
    def __init__(self, parent=None, window=10, n=None, nmax=1000, initval=0.0):
        Plotx.__init__(self, parent)

        if n is not None:
            n = max(int(n), 1)
            nmax = max(nmax, n)
        if window is not None:
            window = float(window)

        initval = float(initval)

        self._yrange = None
        self._window = window
        self._n = n
        if self._n is not None:
            self._sampleIndex = np.arange(self._n - 1, -1, -1)
            self._y = np.tile(initval, [self._n])
        else:
            self._sampleIndex = np.arange(nmax - 1, -1, -1)
            self._y = np.tile(initval, [nmax])

        if self._window is not None:
            self._t = np.tile(-self._window, [nmax])
        else:
            self._t = np.tile(initval, [nmax])

        self._target = None
        if self._window is not None:
            self.plot(self._t, self._y, pen='g', plotid='live')
        else:
            self.plot(self._sampleIndex, self._y, pen='g', plotid='live')
        self.hLine(None, lineid='average', label='{value:0.1f}',
                   labelOpts={'movable':True}, pen='b').setVisible(False)
        self.hLine(None, lineid='target', pen='r').setVisible(False)
        self.hideStatusBar()
        self.hideCursorLine()
        self.getPlotItem().invertX(True)

        self._showAverageCheckbox = QCheckBox()
        self._showAverageCheckbox.setText(
            QCoreApplication.translate('LivePlotx', 'Average'))
        self._showAverageCheckbox.setTristate(False)
        self._showAverageCheckbox.setToolTip(
            QCoreApplication.translate('LivePlotx',
                                       'Show/hide the window average.'))
        self._showAverageCheckbox.stateChanged.connect(
            self.setAverageVisible)

        self.statusBar().export.disconnect()
        self.statusBar().export.connect(self.export)

        self.statusBar().layout().insertWidget(
            LivePlotx.CHECKBOX_START_INDEX, self._showAverageCheckbox)

        if self._window is not None:
            self.setXRange(0.0, self._window, padding=0)

    def export(self, filename=None):
        if filename is None:
            filename = QFileDialog.getSaveFileName(
                self,
                QCoreApplication.translate('LivePlotx', 'Export data to file'),
                '',
                QCoreApplication.translate('LivePlotx',
                                           'Data files (*.mat *.npz)')
            )
            if isinstance(filename, tuple):
                filename = filename[0]

        if filename:
            root, ext = os.path.splitext(filename)
            ext = ext.lower()

            if self._window is not None:
                mask = (self._t[-1] - self._t) <= self._window
                data = self._y[mask]
                t = self._t[mask]
            else:
                data = self._y[-self._n:]
                t = self._t[-self._n:]

            try:
                if ext == '.mat':
                    scipy.io.savemat(
                        filename, {'data':data, 't':t}, do_compression=True)
                elif ext == '.npz':
                    np.savez_compressed(filename, data, t)
            except:
                dlg = dialogs.ErrorMessage(
                    self,
                    QCoreApplication.translate('LivePlotx', 'Export data'),
                    QCoreApplication.translate(
                        'LivePlotx',
                        'Falied to export data to the specified file!'
                    ),
                    traceback.format_exc()
                )
                dlg.exec_()

    def mouseDoubleClickEvent(self, event):

        super().mouseDoubleClickEvent(event)

        if self._window is not None:
            self.setXRange(0.0, self._window, padding=0)
        if self._yrange is not None:
            self._plot.setYRange(*self._yrange)

    def setAverageVisible(self, state):
        self.getHLine('average').setVisible(state)
        self._showAverageCheckbox.setChecked(state)

    def setTargetVisible(self, state):
        self.getHLine('target').setVisible(state)

    def setLivePen(self, pen):
        self.item('live').setPen(pen)

    def setAveragePen(self, pen):
        self.getHLine('average').setPen(pen)

    def setAverageFormat(self, fmt, suffix=None):
        if suffix is None:
            self.getHLine('average').label.setFormat(
                'mean={{value:{}}}'.format(fmt))
        else:
            self.getHLine('average').label.setFormat(
                'mean={{value:{}}} {}'.format(fmt, suffix))

    def setTargetPen(self, pen):
        self.getHLine('target').setPen(pen)

    def setTarget(self, y):
        self.setHLinePosition('target', y)

    def addSample(self, y, t=None):
        if t is None:
            t = time.perf_counter()

        self._t[0:-1] = self._t[1:]
        self._t[-1] = t

        self._y[0:-1] = self._y[1:]
        self._y[-1] = y

        dt = t - self._t
        if self._window is not None:
            mask = dt <= self._window
            self.item('live').setData(dt[mask], self._y[mask])
            if self.getHLine('average').isVisible():
                self.setHLinePosition('average', self._y[mask].mean())
        else:
            self.item('live').setData(self._sampleIndex, self._y[-self._n:])
            if self.getHLine('average').isVisible():
                self.setHLinePosition('average', self._y[-self._n:].mean())

    def addSamples(self, y, t=None):
        if t is None:
            t = time.perf_counter()

        y = np.asarray(y)
        n = y.size

        self._t[0:-n] = self._t[n:]
        self._t[-n:] = t

        self._y[0:-n] = self._y[n:]
        self._y[-n:] = y

        dt = t - self._t
        mask = dt <= self._window
        self.item('live').setData(dt[mask], self._y[mask])
        if self.getHLine('average').isVisible():
            self.setHLinePosition('average', self._y[mask].mean())

    def exportData(self):
        return self.item('live').xData, self.item('live').yData

    def setYRange(self, low, high, *args, **kwargs):
        self._yrange = [low, high]
        self._plot.setYRange(low, high, *args, **kwargs)

if __name__ == '__main__':
    app = common.prepareqt()

    x = np.linspace(0, np.pi, 1000)
    spectrum = np.sin(x)

    pltx = Plotx()
    pltx.plot(x, spectrum*1e9)
    pltx.show()

    app.exec_()

    '''
    lines = [400, 500, 600, 700, 800, 900]
    plt = Plot()
    plt.plot(spectrum)
    for line in lines:
        movable = line in [lines[0], lines[-1]]
        plt.vLine(line, movable=movable, label='{value:.1f}'),
    sb = StatusBar(plt)
    plt.show()
    sb.show()
    plt.autoRange()
    '''
