# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 22:51:01 2017

@author: miran
"""
import sys
import os.path
import traceback

import numpy as np
from scipy.io import savemat

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
                            QLabel, QCheckBox, QSizePolicy, \
                            QSplitter, QFileDialog, QSlider
from PyQt5.QtGui import QColor, QCursor, QMouseEvent
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtCore import pyqtSignal as QSignal

import pyqtgraph as pg
from PIL import Image as pilimage
from widgets import dialogs, common, plot

class _HistogramLUTWidget(pg.HistogramLUTWidget):
    doubleClicked = QSignal(QMouseEvent)
    def __init__(self, *args, **kwargs):
        pg.HistogramLUTWidget.__init__(self, *args, **kwargs)

    def mouseDoubleClickEvent(self, event):
        event.accept()
        self.doubleClicked.emit(event)
        pg.HistogramLUTWidget.mouseDoubleClickEvent(self, event)

class _GraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    doubleClicked = QSignal(QMouseEvent)
    def __init__(self, *args, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, *args, **kwargs)

    def mouseDoubleClickEvent(self, event):
        event.accept()
        self.doubleClicked.emit(event)
        pg.HistogramLUTWidget.mouseDoubleClickEvent(self, event)

class ImageStatusBar(QWidget):
    sigFrozen = QSignal(bool)
    export = QSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        hLayout = QHBoxLayout()
        self._width = self._height = self._depth = None
        hLayout.setContentsMargins(5, 5, 5, 5)

        self._positionLabel = QLabel()
        self._intensityLabel = QLabel()
        self._infoLabel = QLabel()
        self._sizeLabel = QLabel()
        self._sectiomMarkerCheckBox = QCheckBox(
            QCoreApplication.translate('ImageStatusBar', 'Profiles'))
        self._sectiomMarkerCheckBox.setToolTip(
            QCoreApplication.translate(
                'ImageStatusBar',
                'Show/hide horizontal and vertical profiles.'))
        self._sectiomMarkerCheckBox.setTristate(False)
        self._sectiomMarkerCheckBox.setCheckState(Qt.Checked)

        self._titleLabel = QLabel('')

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
                'Export image to a file.'))

        hLayout.addWidget(self._infoLabel)
        hLayout.addWidget(self._sizeLabel)
        hLayout.addWidget(self._positionLabel)
        hLayout.addWidget(self._intensityLabel)
        hLayout.addStretch(0)
        hLayout.addWidget(self._titleLabel)
        hLayout.addStretch(0)
        hLayout.addWidget(self._sectiomMarkerCheckBox)
        hLayout.addWidget(self._frozenCheckBox)
        hLayout.addWidget(self._exportPushButton)

        self.setLayout(hLayout)
        self.sizePolicy().setVerticalPolicy(QSizePolicy.Minimum)

    def _frozenChanged(self, value):
        self.sigFrozen.emit(value)

    def setFrozen(self, value):
        self._frozenCheckBox.setChecked(value)

    def frozen(self):
        return bool(self._frozenCheckBox.isChecked())

    def setIntensity(self, intensity, typename=''):
        if intensity is None:
            self._intensityLabel.clear()
        else:
            label_str = ''
            if np.issubdtype(intensity.dtype, np.integer):
                if intensity.size == 1:
                    format_str = '{}: {:<6}'
                    label_str = format_str.format(typename, intensity)
                else:
                    format_str = '{}: ' + ', '.join(['{:<6}']*intensity.size)
                    label_str = format_str.format(typename, *intensity)
            else:
                if intensity.size == 1:
                    format_str = '{}: {:<6.3f}'
                    label_str = format_str.format(typename, intensity)
                else:
                    format_str = '{}: ' + ', '.join(['{:<6.3f}']*intensity.size)
                    label_str = format_str.format(typename, *intensity)
            self._intensityLabel.setText(label_str)

    def setImageSize(self, width, height, depth=None):
        if height != self._height or width != self._width or \
                depth != self._depth:
            if depth is None:
                sizeText = '({} x {})'.format(width, height)
            else:
                sizeText = '({} x {} x {})'.format(
                    width, height, depth)
            self._sizeLabel.setText(sizeText)

    def setImageInfo(self, info):
        if info is None:
            self._infoLabel.clear()
        else:
            self._infoLabel.setText(info)

    def setPosition(self, pos):
        self._positionLabel.setText('x={:<6.1f}, y={:<6.1f}'.format(
            pos.x(), pos.y()))

    def setTitle(self, title):
        self._titleLabel.setText(title)

    def title(self):
        return self._titleLabel.text()

class ImageSlicer(QWidget):
    requestImageUpdate = QSignal(int)

    def __init__(self, parent=None):

        super().__init__(parent)

        self._horizontalSlider = QSlider(Qt.Horizontal)
        self._horizontalSlider.setSingleStep(1)
        self._horizontalSlider.setMinimum(0)
        self._horizontalSlider.valueChanged.connect(self.sendRequestImageUpdate)

        self._label = QLabel()
        self._label.setMaximumWidth(200)
        self._label.setMinimumWidth(100)

        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self._horizontalSlider)
        layout.addWidget(self._label)

        self.setLayout(layout)
        self.sizePolicy().setVerticalPolicy(QSizePolicy.Minimum)

    def setSliderPts(self, pts):
        self._horizontalSlider.setMaximum(pts-1)

    def setSliderLabel(self, label):
        self._label.setText(label)

    def sliderValue(self):
        return self._horizontalSlider.value()

    def setSliderValue(self, value):
        self._horizontalSlider.setValue(value)

    def sendRequestImageUpdate(self):
        self.requestImageUpdate.emit(self._horizontalSlider.value())

class Image(QWidget):

    def __init__(self, data=None, levels=None, range=None,
                 title=None, parent=None, sliderange=None):

        QWidget.__init__(self, parent)

        if title is None:
            title = QCoreApplication.translate('Image', 'Image')

        self._frozen = False
        self._data = self._range = None

        self._profile_pans = [pg.mkPen(item) for item in 'rgbk']

        self.setWindowTitle(title)
        self._hplotx = self._vploty = None
        self._hplotpos = self._vplotpos = None
        vLayout = QVBoxLayout()
        vLayout.setContentsMargins(0, 0, 0, 0)
        vLayout.setSpacing(0)

        self._hsplitter = QSplitter(Qt.Horizontal)
        self._hsplitter.setStretchFactor(0, 0)
        self._hsplitter.setStretchFactor(0, 1)
        self._hsplitter.setStretchFactor(0, 0)
        self._data = self._range = None

        self._graphicsWidget = _GraphicsLayoutWidget()
        self._imageViewBox = self._graphicsWidget.addViewBox(enableMenu=False)
        self._pgimage = pg.ImageItem(axisOrder='row-major')
        self._imageViewBox.addItem(self._pgimage)

        self._histogramLUTWidget = _HistogramLUTWidget(image=self._pgimage)
        self._histogramLUTWidget.item.vb.setMenuEnabled(False)
        self._histogramLUTWidget.item.vb.setMouseEnabled(False, False)
        self._histogramLUTWidget.doubleClicked.connect(self._doubleClickedEvent)
        self._histogramLUTWidget.setMinimumWidth(
            int(self._histogramLUTWidget.item.minimumWidth()))
        self._histogramLUTWidget.setMaximumWidth(
            int(self._histogramLUTWidget.item.maximumWidth()))

        self._hplot = plot.Plot()
        self._hplot.setMaximumHeight(
            int(self._histogramLUTWidget.item.maximumWidth()))
        self._hplotpos = self._hplot.plotItem.addLine(
            x=0,
            pen=pg.mkPen(color=QColor(Qt.red)))
        self._vplot = plot.Plot()
        self._vplot.setMaximumWidth(
            int(self._histogramLUTWidget.item.maximumWidth()))
        self._vplotpos = self._vplot.plotItem.addLine(
            y=0,
            pen=pg.mkPen(color=QColor(Qt.red)))
        
        self._imageSlicer = ImageSlicer()
        self._imageSlicer.setMaximumHeight(
            self._imageSlicer.minimumSizeHint().height())
        self._imageSlicer.hide()

        self._imageViewBox.setAspectLocked(True)
        self._statusBar = ImageStatusBar()
        self._statusBar.sigFrozen.connect(self.setFrozen)
        self._statusBar.setMaximumHeight(
            self._statusBar.minimumSizeHint().height())
        self._statusBar._sectiomMarkerCheckBox.stateChanged.connect(
            self._showHideLines)
        self._statusBar.export.connect(self._exportImage)

        self._hsplitter.addWidget(self._vplot)
        self._hsplitter.addWidget(self._graphicsWidget)
        self._hsplitter.addWidget(self._histogramLUTWidget)

        self._vsplitter = QSplitter(Qt.Vertical)
        self._vsplitter.setStretchFactor(0, 0)
        self._vsplitter.setStretchFactor(0, 1)
        self._vsplitter.setStretchFactor(0, 0)
        self._vsplitter.addWidget(self._hplot)
        self._vsplitter.addWidget(self._hsplitter)
        self._vsplitter.addWidget(self._imageSlicer)
        self._vsplitter.addWidget(self._statusBar)
        vLayout.addWidget(self._vsplitter, stretch=1)

        self._graphicsWidget.scene().sigMouseMoved.connect(
            self._mouseMove)
        self._graphicsWidget.scene().sigMouseClicked.connect(
            self._fitMouseEvent)
        self._graphicsWidget.doubleClicked.connect(lambda event: self.fit())

        if levels is not None:
            self.setLevels(levels)
        if range is not None:
            self.setRange(range)

        self._hline = pg.InfiniteLine(0, 0, movable=True, label='{value:.1f}',
                                      labelOpts={'position':0.05})
        self._hline.sigPositionChanged.connect(self._updateHline)

        self._vline = pg.InfiniteLine(0, 90, movable=True, label='{value:.1f}',
                                      labelOpts={'position':0.95})
        self._vline.sigPositionChanged.connect(self._updateVline)

        self._imageViewBox.addItem(self._hline)
        self._imageViewBox.addItem(self._vline)

        self._roi_rect = None
        self._crosshairs = {}

        self.update()

        self.setImage(data)

        self.setLayout(vLayout)
        self._graphicsWidget.show()

    def _exportImage(self):
        if self._data is None:
            dialogs.InformationMessage(
                self,
                QCoreApplication.translate('Image', 'Export image'),
                QCoreApplication.translate('Image', 'Nothing to export')
            )
        else:
            filename = QFileDialog.getSaveFileName(
                self,
                QCoreApplication.translate('Image', 'Export image to a file'),
                None,
                QCoreApplication.translate(
                    'Image',
                    'JPG image file (*.jpg);;'\
                    'PNG image file (*.png);;BMP image file (*.bmp);;'\
                    'Numpy NPZ data file (*.npz);;Matlab data file (*.mat);;'
                )
            )
            if isinstance(filename, tuple):
                filename = filename[0]

            if filename:
                try:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext == '.npz':
                        np.savez_compressed(filename, data=self._data)
                    elif ext == '.mat':
                        savemat(filename, {'data':self._data})
                    else:
                        if self._data.dtype != np.uint8:
                            data = self._data.astype(np.float64)* \
                                (255.0/self._data.max()) + 0.5
                            data = data.astype(np.uint8)
                        else:
                            data = self._data

                        if filename:
                            pilimage.fromarray(data).save(filename)
                except:
                    dlg = dialogs.ErrorMessage(
                        self,
                        QCoreApplication.translate('Image', 'Export image'),
                        QCoreApplication.translate(
                            'Image',
                            'Falied to export image to the specified file!'),
                        traceback.format_exc()
                    )
                    dlg.exec_()

    def _updateHline(self):
        pos = self._hline.getYPos()
        if self._data is not None and \
                self._statusBar._sectiomMarkerCheckBox.checkState() == \
                    Qt.Checked:
            index = np.clip(int(round(pos)), 0, self._data.shape[0] - 1)
            if self._data.ndim == 2:
                self._hplot.setData('hline_0',
                                    y=self._data[index, :], x=self._hplotx)
                #self.removePlot('hline_1')
                #self.removePlot('hline_2')
            else:
                for channel in range(self._data.shape[2]):
                    self._hplot.setData(
                        'hline_{:d}'.format(channel),
                        y=self._data[index, :, channel], x=self._hplotx,
                        pen=self._profile_pans[channel]
                    )
            self._vplotpos.setValue(index)

    def _updateVline(self):
        pos = self._vline.getXPos()
        if self._data is not None and \
                self._statusBar._sectiomMarkerCheckBox.checkState() == \
                    Qt.Checked:
            index = np.clip(int(round(pos)), 0, self._data.shape[1] - 1)
            if self._data.ndim == 2:
                self._vplot.setData('vline_0',
                                    x=self._data[:, index], y=self._vploty)
                #self.removePlot('vline_1')
                #self.removePlot('vline_2')
            else:
                for channel in range(self._data.shape[2]):
                    self._vplot.setData(
                        'vline_{:d}'.format(channel),
                        x=self._data[:, index, channel], y=self._vploty,
                        pen=self._profile_pans[channel]
                    )

            self._hplotpos.setValue(index)

    def _showHideLines(self, state):
        if state == Qt.Checked:
            state = True
        else:
            state = False
        self._hline.setVisible(state)
        self._vline.setVisible(state)

        self._hplot.setVisible(state)
        self._vplot.setVisible(state)
        self._statusBar._sectiomMarkerCheckBox.setChecked(state)

    def showProfiles(self):
        self._showHideLines(True)

    def hideProfiles(self):
        self._showHideLines(False)

    def setFrozen(self, state):
        self._frozen = bool(state)
        self._statusBar.setFrozen(state)

    def frozen(self):
        return self._frozen

    def setImage(self, data, info=None, **kwargs):
        if data is not None and not self._frozen:
            self._data = data
            self._pgimage.setImage(data, levels=self.levels(), **kwargs)
            if self._range is None:
                self._range = self._pgimage.quickMinMax()
                self.setLevels(self._range)
            pos = self._graphicsWidget.mapFromGlobal(QCursor.pos())
            self._mouseMove(pos)

            self._hline.setBounds([0, data.shape[0]])
            if self._hplotx is None or self._hplotx.size != data.shape[1]:
                self._hplotx = np.arange(data.shape[1])
            self._vline.setBounds([0, data.shape[1]])
            if self._vploty is None or self._vploty.size != data.shape[0]:
                self._vploty = np.arange(data.shape[0])

            self._statusBar.setImageSize(data.shape[1], data.shape[0])
            self._statusBar.setImageInfo(info)
            self._updateVline()
            self._updateHline()

    def fit(self):
        self._imageViewBox.autoRange(padding=0.005)

    def _fitMouseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.fit()

    def _checkLevels(self, low, high):
        if self._range is not None:
            low = np.clip(low, *self._range)
            high = np.clip(high, *self._range)

    def _doubleClickedEvent(self, event):
        if Qt.ControlModifier & event.modifiers():
            if self._data is not None:
                self.setLevels(self._pgimage.quickMinMax())
        else:
            self.resetLevels()

    def resetLevels(self):
        if self._range is not None:
            self._histogramLUTWidget.setLevels(*self._range)

    def setLevels(self, levels):
        self._histogramLUTWidget.item.setLevels(*levels)

    def levels(self):
        #return self._pgimage.getLevels()
        return self._histogramLUTWidget.item.getLevels()

    def autoRange(self):
        self._histogramLUTWidget.autoHistogramRange()

    def setRange(self, range):
        if range is None:
            self.autoRange()
        else:
            self._range = np.asarray(range)
            self._histogramLUTWidget.setHistogramRange(*range)

    def range(self):
        return self._range

    def histogram(self, bins=128, **kwargs):
        return self._pgimage.getHistogram(bins, range=self._range, **kwargs)

    def _mouseMove(self, pos):
        try:
            if self._data is not None:
                imgpos = self._pgimage.mapFromScene(pos)
                x, y = imgpos.x(), imgpos.y()
                if x > 0 and x < self._data.shape[1] and \
                        y > 0 and y < self._data.shape[0]:
                    self._statusBar.setIntensity(self._data[int(y), int(x)],
                                                 self._data.dtype.name)
                else:
                    self._statusBar.setIntensity(None)

                self._statusBar.setPosition(imgpos)
        except:
            print(sys.exc_info())

    def hideStatusBar(self):
        #self._statusBar.hide()
        self._vsplitter.setSizes((1, 0))

    def showStatusBar(self):
        #self._statusBar.show()
        self._vsplitter.setSizes((1, 1))

    def collapseStatusBar(self):
        siz = self._vsplitter.sizes()
        self._vsplitter.setSizes((siz[0] + siz[1], 0,))

    def hideHistogram(self):
        self._histogramLUTWidget.hide()

    def showHistogram(self):
        self._histogramLUTWidget.show()

    def collapseHistogram(self):
        siz = im._hsplitter.sizes()
        im._hsplitter.setSizes((siz[0] + siz[1], 0))

    def _mouseClick(self, event):
        if event.double() and event.button() == Qt.LeftButton:
            self.fit()

    def data(self):
        return self._data

    def imageItem(self):
        return self._pgimage

    def addItem(self, item):
        self._imageViewBox.addItem(item)

    def removeItem(self, item):
        self._imageViewBox.removeItem(item)

    def setTitle(self, title):
        self._statusBar.setTitle(title)

    def title(self):
        return self._statusBar.title()
    
    def toggleroi(self, switch: bool, pos: list, size: list, color: str):
        if switch:
            self._roi_rect = pg.RectROI(pos, size, pen=pg.mkPen(color=color))
            self.addItem(self._roi_rect)
        else:
            self.removeItem(self._roi_rect)
            self._roi_rect = None

    def togglecrosshair(self, key, switch: bool, pos: list, color: str):
        if switch:
            if key in self._crosshairs.keys():
                self.removeItem(self._crosshairs[key])
            self._crosshairs[key] = pg.CrosshairROI(
                pos, pen=pg.mkPen(color=color), size=(20,20), angle=45,
                movable=False, rotatable=False, resizable=False)
            self.addItem(self._crosshairs[key])
        elif key in self._crosshairs.keys():
            self.removeItem(self._crosshairs[key])
            self._crosshairs.pop(key)

if __name__ == '__main__':
    '''
    pg.mkQApp()

    w = pg.GraphicsLayoutWidget()
    w.show()
    vb = w.addViewBox()
    img = pg.ImageItem(np.random.normal(size=(100,100)))

    rect = QGraphicsRectItem(10,10, 50, 25)
    rect.setPen(QPen(QBrush(Qt.red),1.0))

    line = QGraphicsLineItem(0, 0, 100, 100)
    line.setPen(QPen(QBrush(Qt.red),1.0))

    vb.setLimits(xMin=0, xMax=100, yMin=0, yMax=100,
                 minXRange=1, maxXRange=100)
    vb.addItem(img)
    vb.addItem(line)
    vb.addItem(rect)

    def mouseMoved(pos):
        imgpos = img.mapFromScene(pos)
        sb.setPosition(imgpos)
        print("Image position:", imgpos)

    w.scene().sigMouseMoved.connect(mouseMoved)
    '''
    from widgets import common
    
    app = common.prepareqt()

    shape = (1024, 1920, 3)
    data = np.random.randint(0, 64, size=shape).astype(np.uint8)
    im = Image()
    im.hideProfiles()
    im.setImage(data)
    im.setLevels((0, 1,))
    #im.setLevels((0, 128,))
    timer = pg.QtCore.QTimer(parent=im)
    #img = np.random.rand(*shape)
    def update():
        im.setImage(np.random.rand(*shape))

    timer.timeout.connect(update)
    timer.setInterval(int(1000/10))
    timer.start()
    im.show()

    app.exec()
    timer.stop()
    #l=pg.InfiniteLine(960, 90, movable=True)
    #l.setBounds([0,1920-1])
    #im._imageViewBox.addItem(l)

    '''
    timer = pg.QtCore.QTimer()
    def update():
        im.setImage(np.random.normal(size=(200,100)))
    timer.timeout.connect(update)
    timer.setInterval(100)
    timer.start()
    '''
