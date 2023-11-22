import os
import sys
from warnings import warn
import copy
import time

from PyQt5 import QtCore, QtGui, QtWidgets, uic

import pyqtgraph as pg
import numpy as np
from PIL import Image

#sys.path.append(".")
import clinfo
from angularspectrum import Propagator

# define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'ui', 'mainReconstruction.ui')

WindowTemplate, TemplateBaseClass = uic.loadUiType(uiFile)

class CustomImageItem(pg.ImageItem):

    sendDataSignal = QtCore.pyqtSignal(dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setAcceptHoverEvents(True)

    def hoverMoveEvent(self, ev):
        self.sendDataSignal.emit(
            {
                'pos': (int(ev.pos().x()), int(ev.pos().y())),
                'data': self.image[int(ev.pos().y()), int(ev.pos().x())]
            }
        )

class CustomViewBox(pg.ViewBox):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mouseDoubleClickEvent(self, ev):
        self.autoRange()

class CustomHistogramLUTItem(pg.HistogramLUTItem):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mouseDoubleClickEvent(self, ev):
        self.vb.autoRange()

class MainWindow(TemplateBaseClass):

    send_termination = QtCore.pyqtSignal(bool)

    def __init__(self):
        TemplateBaseClass.__init__(self)
        
        # create the main window
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        self.ui.checkBoxROI.toggled.connect(self.toggleROIrecon)
        self.ui.checkBoxBackgroundROI.toggled.connect(self.toggleROIbackground)
        self.viewbox = CustomViewBox(invertY=True, lockAspect=True)
        self.ui.graphicsView.setCentralWidget(self.viewbox)
        self.imgitem = CustomImageItem()
        self.imgitem.axisOrder = 'row-major'
        self.viewbox.addItem(self.imgitem)
        self.ui.graphicsViewHistogram.setCentralWidget(CustomHistogramLUTItem(image=self.imgitem))

        # functionality with the data points
        self.imgitem.sendDataSignal.connect(self.setCoorsValues)

        # functionality with the files
        self._fileHologram = None
        self._fileBackground = None
        self._fileDark = None
        self.ui.pushButtonLoadHologram.clicked.connect(lambda: self.loadFile('Hologram'))
        self.ui.pushButtonLoadBackground.clicked.connect(lambda: self.loadFile('Background'))
        self.ui.pushButtonLoadDark.clicked.connect(lambda: self.loadFile('Dark'))

        self.ui.checkBoxHologram.toggled.connect(lambda: self.deleteFile('Hologram'))
        self.ui.checkBoxBackground.toggled.connect(lambda: self.deleteFile('Background'))
        self.ui.checkBoxDark.toggled.connect(lambda: self.deleteFile('Dark'))

        self.ui.checkBoxSqHologram.toggled.connect(self.operationFile)
        self.ui.checkBoxDivideBackground.toggled.connect(self.operationFile)
        self.ui.checkBoxSubtractDark.toggled.connect(self.operationFile)
        self.ui.checkBoxDivideROI.toggled.connect(self.operationFile)

        # GPU functionality
        self.ui.checkBoxGPU.toggled.connect(self.displayDevices)
        self.ui.comboGPU.activated.connect(self.selectDevice)

        # single backpropagation functionality
        self.devices = None
        self.idevice = None
        self._terminate_propagation_thread = False
        self.ui.pushButton_propagate.clicked.connect(self.backPropagation)

        # reconstruction funcionality
        self._terminate_reconstruction_thread = False
        self.ui.pushButton_reconstruct.clicked.connect(self.iterativeReconstruction)

        self.roi_background = None
        self.roi_recon = None
        self.threadpool = QtCore.QThreadPool()
        self.show()

    def displayDevices(self):
        if self.ui.checkBoxGPU.isChecked():
            self.devices = clinfo.gpus()
            for device in self.devices:
                self.ui.comboGPU.addItem(device.name)
        else:
            self.devices = None
            self.idevice = None
            self.ui.comboGPU.clear()

    def selectDevice(self, index):
        self.idevice = index

    def backPropagation(self):

        if self._fileHologram is None:
            return

        if not self.check_parameters():
            return

        if not self.check_propagation_parameters():
            return

        self.ui.pushButton_propagate.clicked.disconnect()
        self.ui.pushButton_propagate.setText('Cancel')
        self.ui.pushButton_propagate.clicked.connect(self.terminate_thread)

        if self.roi_recon is not None:
            x1, y1, x2, y2 = self.getroi(self.imgitem.image, self.roi_recon)

            if x2 > x1 and y2 > y1:
                hologramROI = self.imgitem.image[y1:y2, x1:x2]
        else:
            hologramROI = self.imgitem.image

        pixelsize = 1e-6 * float(self.ui.lineEdit_sensor_pixelsize.text().replace(',','.'))
        magnification = float(self.ui.lineEdit_magnification.text().replace(',','.'))
        wavelength = 1e-9 * float(self.ui.lineEdit_wavelength.text().replace(',','.'))
        z_min = 1e-6 * float(self.ui.lineEdit_minz_propagation.text().replace(',','.'))
        z_max = 1e-6 * float(self.ui.lineEdit_maxz_propagation.text().replace(',','.'))
        n_steps = int(self.ui.lineEdit_steps_propagation.text())

        z = np.linspace(z_min, z_max, n_steps)

        if self.devices is not None and self.idevice is not None:
            self.backpropagator = BackPropagatorThread(hologramROI, np.zeros_like(hologramROI),
                pixelsize/magnification, wavelength, z, self.send_termination, self.devices[self.idevice])
        else:
            self.backpropagator = BackPropagatorThread(hologramROI, np.zeros_like(hologramROI),
                pixelsize/magnification, wavelength, z, self.send_termination)

        self.backpropagator.signals.progress.connect(self.updateProgressPropagation)
        self.backpropagator.signals.finished.connect(lambda x, y: self.openSlicer(z, x, y))
        self.backpropagator.signals.check_termination.connect(self.check_termination)
        self.backpropagator.start()

    def terminate_thread(self):
        self._terminate_propagation_thread = True
        self._terminate_reconstruction_thread = True

    def check_termination(self):
        if self._terminate_propagation_thread or self._terminate_reconstruction_thread:
            self.send_termination.emit(True)
        
    def updateProgressPropagation(self, i: int):
        self.ui.progress_propagation.setValue(i)

    def iterativeReconstruction(self):

        if self._fileHologram is None:
            return

        if not self.check_parameters():
            return

        if not self.check_reconstruction_parameters():
            return

        self.ui.pushButton_reconstruct.clicked.disconnect()
        self.ui.pushButton_reconstruct.setText('Cancel')
        self.ui.pushButton_reconstruct.clicked.connect(self.terminate_thread)

        if self.roi_recon is not None:
            x1, y1, x2, y2 = self.getroi(self.imgitem.image, self.roi_recon)

            if x2 > x1 and y2 > y1:
                hologramROI = self.imgitem.image[y1:y2, x1:x2]
        else:
            hologramROI = self.imgitem.image

        pixelsize = 1e-6 * float(self.ui.lineEdit_sensor_pixelsize.text().replace(',','.'))
        magnification = float(self.ui.lineEdit_magnification.text().replace(',','.'))
        wavelength = 1e-9 * float(self.ui.lineEdit_wavelength.text().replace(',','.'))
        z_min = 1e-6 * float(self.ui.lineEdit_minz_reconstruction.text().replace(',','.'))
        z_max = 1e-6 * float(self.ui.lineEdit_maxz_reconstruction.text().replace(',','.'))
        n_steps = int(self.ui.lineEdit_steps_reconstruction.text())
        iter_num = int(self.ui.lineEdit_iter_reconstruction.text())

        z = np.linspace(z_min, z_max, n_steps)

        if self.devices is not None and self.idevice is not None:
            self.reconstructor = IteratorThread(hologramROI, np.zeros_like(hologramROI),
                pixelsize/magnification, wavelength, z, iter_num, self.send_termination, self.devices[self.idevice])
        else:
            self.reconstructor = IteratorThread(hologramROI, np.zeros_like(hologramROI),
                pixelsize/magnification, wavelength, z, iter_num, self.send_termination)

        self.reconstructor.signals.progress.connect(self.updateProgressReconstruction)
        self.reconstructor.signals.finished.connect(lambda x, y: self.openSlicer(z, x, y))
        self.reconstructor.signals.check_termination.connect(self.check_termination)
        self.reconstructor.start()

    def updateProgressReconstruction(self, i: int):
        self.ui.progress_reconstruction.setValue(i)

    def openSlicer(self, z, amplitudes, phases):

        self.ui.pushButton_propagate.clicked.disconnect()
        self.ui.pushButton_propagate.setText('Propagate')
        self.ui.pushButton_propagate.clicked.connect(self.backPropagation)
        self.ui.progress_propagation.setValue(0)

        self.ui.pushButton_reconstruct.clicked.disconnect()
        self.ui.pushButton_reconstruct.setText('Reconstruct')
        self.ui.pushButton_reconstruct.clicked.connect(self.iterativeReconstruction)
        self.ui.progress_reconstruction.setValue(0)

        if not self._terminate_propagation_thread and \
            not self._terminate_reconstruction_thread:

            self.slicer = SlicerWindow(z, amplitudes, phases)

        self._terminate_propagation_thread = False
        self._terminate_reconstruction_thread = False

    def setCoorsValues(self, data):
        self.ui.labelValue.setText('{:10.3f}'.format(data['data']))
        self.ui.labelXYcoor.setText('x ={:9.3f}, y ={:9.3f}'.format(data['pos'][0], data['pos'][1]))

    def loadFile(self, descr):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image files (*.png *.jpg *.bmp *.tif *.tiff);; NPZ files (*.npz)")
        
        if fileName[0]:
            getattr(self.ui, 'pushButtonLoad' +  descr).setEnabled(False)
            getattr(self.ui, 'pushButtonLoad' +  descr).setText(descr + ' loaded')
            getattr(self.ui, 'checkBox' +  descr).setCheckState(QtCore.Qt.Checked)
            getattr(self.ui, 'checkBox' +  descr).setEnabled(True)
            if fileName[1] ==  'Image files (*.png *.jpg *.bmp *.tif *.tiff)':
                setattr(self, '_file' + descr, np.array(Image.open(fileName[0])))
            elif fileName[1] == 'NPZ files (*.npz)':
                setattr(self, '_file' + descr, np.load(fileName[0])['sample'])

            if descr == 'Hologram':
                self.imgitem.setImage(getattr(self, '_file' + descr))

    def deleteFile(self, descr):
        if not getattr(self.ui, 'checkBox' +  descr).isChecked():
            getattr(self.ui, 'pushButtonLoad' +  descr).setEnabled(True)
            getattr(self.ui, 'pushButtonLoad' +  descr).setText('Load ' + descr)
            getattr(self.ui, 'checkBox' +  descr).setCheckState(QtCore.Qt.Unchecked)
            getattr(self.ui, 'checkBox' +  descr).setEnabled(False)
            setattr(self, '_file' + descr, None)
        
        if descr == 'Hologram':
            self.imgitem.clear()
            self.ui.checkBoxSqHologram.setCheckState(QtCore.Qt.Unchecked)
            self.ui.checkBoxSqHologram.setCheckState(QtCore.Qt.Unchecked)

    def operationFile(self):

        if self._fileHologram is None:
            print('Hologram is not loaded')
            self.ui.checkBoxSubtractDark.setCheckState(QtCore.Qt.Unchecked)
            self.ui.checkBoxDivideBackground.setCheckState(QtCore.Qt.Unchecked)
            self.ui.checkBoxSqHologram.setCheckState(QtCore.Qt.Unchecked)
            self.ui.checkBoxDivideROI.setCheckState(QtCore.Qt.Unchecked)
            return
        else:
            hologram = self._fileHologram

        if self.ui.checkBoxSubtractDark.isChecked():
            if self._fileDark is None:
                print('Dark is not loaded')
                self.ui.checkBoxSubtractDark.setCheckState(QtCore.Qt.Unchecked)
            else:
                hologram = np.abs(hologram - self._fileDark)
                hologram[hologram < 0] = 0
        
        if self.ui.checkBoxDivideBackground.isChecked():
            if self._fileBackground is None:
                print('Background is not loaded')
                self.ui.checkBoxDivideBackground.setCheckState(QtCore.Qt.Unchecked)
            else:
                if self.ui.checkBoxSubtractDark.isChecked():
                    background = np.abs(self._fileBackground - self._fileDark)
                else:
                    background = self._fileBackground
                self.ui.checkBoxDivideROI.setCheckState(QtCore.Qt.Unchecked)
                hologram = hologram/background
                hologram[np.isinf(hologram)] = 0

        if self.ui.checkBoxDivideROI.isChecked() and \
            not self.ui.checkBoxDivideBackground.isChecked() and \
            self.roi_background is not None:

            x1, y1, x2, y2 = self.getroi(hologram, self.roi_background)

            if x2 > x1 and y2 > y1:
                average = np.mean(hologram[y1:y2, x1:x2])
                hologram = hologram/average
            else:
                self.ui.checkBoxDivideROI.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.ui.checkBoxDivideROI.setCheckState(QtCore.Qt.Unchecked)

        if self.ui.checkBoxSqHologram.isChecked():
            hologram = np.sqrt(hologram)
            hologram[np.isnan(hologram)] = 0.0

        self.imgitem.setImage(hologram)
            
    def toggleROIbackground(self, checked: bool):
        if checked:
            self.roi_background = pg.RectROI((0,0), (100, 100), pen=pg.mkPen(color='b'))
            self.viewbox.addItem(self.roi_background)
        else:
            self.viewbox.removeItem(self.roi_background)
            self.roi_background = None

    def toggleROIrecon(self, checked: bool):
        if checked:
            self.roi_recon = pg.RectROI((0,0), (100, 100), pen=pg.mkPen(color='g'))
            self.viewbox.addItem(self.roi_recon)
        else:
            self.viewbox.removeItem(self.roi_recon)
            self.roi_recon = None

    def check_parameters(self):

        return self.ui.lineEdit_magnification.text() and \
            self.ui.lineEdit_sensor_pixelsize.text() and \
                self.ui.lineEdit_wavelength.text()

    def check_propagation_parameters(self):

        return self.ui.lineEdit_minz_propagation.text() and \
            self.ui.lineEdit_maxz_propagation.text() and \
                self.ui.lineEdit_steps_propagation.text()

    def check_reconstruction_parameters(self):

        return self.ui.lineEdit_minz_reconstruction.text() and \
            self.ui.lineEdit_maxz_reconstruction.text() and \
                self.ui.lineEdit_steps_reconstruction.text() and \
                    self.ui.lineEdit_iter_reconstruction.text()

    def getroi(self, image, roi):

        x1, y1 = roi.pos()
        x1, y1 = max(int(x1), 0), max(int(y1), 0)

        w, h = roi.size()
        x2, y2 = min(int(w) + x1, image.shape[1]), \
            min(int(h) + y1, image.shape[0])

        return x1, y1, x2, y2

path = os.path.dirname(os.path.abspath(__file__))
uiFileSlicer = os.path.join(path, 'ui', 'popupSlicer.ui')

SlicerWindowTemplate, SlicerTemplateBaseClass = uic.loadUiType(uiFileSlicer)

class BackPropagatorSignals(QtCore.QObject):

    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    check_termination = QtCore.pyqtSignal()

class BackPropagator(QtCore.QRunnable):

    def __init__(self, amplitude, phase, pixelsize, wavelength, z, device=None, **kwargs):
        super().__init__()
        self.signals = BackPropagatorSignals()
        self.dtype = np.complex64
        self.propagator = Propagator(amplitude, phase, pixelsize, wavelength, dtype=self.dtype, **kwargs)
        self.device = device
        self.z = z
        self.amplitudes = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self.phases = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)

    @QtCore.pyqtSlot()
    def run(self):

        if self.device is not None:
            self.propagator.init_gpu(cldevices=[self.device,])

        dz = np.zeros_like(self.z)
        dz[1:] = self.z[:-1]
        dz = self.z - dz

        for i, dzi in enumerate(dz):
            field = self.propagator.backpropagate_sequential(dzi)
            self.amplitudes[...,i] = np.abs(field)
            self.phases[...,i] = self.processPhases(np.angle(field))
            self.signals.progress.emit(int(100*(i+1)/dz.size))

        self.signals.finished.emit(self.amplitudes, self.phases)

    def processPhases(self, phase):
        hist, bin_edges = np.histogram(phase, bins=int(np.sqrt(phase.size)))
        centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
        x0 = centers[np.argmax(hist)]

        y = phase - x0
        y[y > np.pi] -= 2*np.pi
        y[y < -np.pi] += 2*np.pi

        return y

class BackPropagatorThread(QtCore.QThread):
    def __init__(self, amplitude, phase, pixelsize, wavelength, z, receive_termination, device=None, **kwargs):
        super().__init__()
        self.signals = BackPropagatorSignals()
        self.dtype = np.complex64
        self.propagator = Propagator(amplitude, phase, pixelsize, wavelength, dtype=self.dtype, **kwargs)
        self.device = device
        self.z = z
        self.amplitudes = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self.phases = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self._terminate = False

        # connect to the termination check
        receive_termination.connect(self.termination_check)

    def run(self):

        if self.device is not None:
            self.propagator.init_gpu(cldevices=[self.device,], clbuild_options=['-cl-fast-relaxed-math', '-cl-mad-enable'])

        dz = np.zeros_like(self.z)
        dz[1:] = self.z[:-1]
        dz = self.z - dz

        for i, dzi in enumerate(dz):
            self.signals.check_termination.emit()
            if self._terminate:
                print('Received the break signal.')
                break
            field = self.propagator.backpropagate_sequential(dzi)
            self.amplitudes[...,i] = np.abs(field)
            self.phases[...,i] = self.processPhases(np.angle(field))
            self.signals.progress.emit(int(100*(i+1)/dz.size))
  
        self.signals.finished.emit(self.amplitudes, self.phases)

    def processPhases(self, phase):
        hist, bin_edges = np.histogram(phase, bins=int(np.sqrt(phase.size)))
        centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
        x0 = centers[np.argmax(hist)]

        y = phase - x0
        y[y > np.pi] -= 2*np.pi
        y[y < -np.pi] += 2*np.pi

        return y

    def termination_check(self, termination):
        self._terminate = termination

class Iterator(QtCore.QRunnable):

    def __init__(self, amplitude, phase, pixelsize, wavelength, z, iter_num, device=None, **kwargs):
        super().__init__()
        self.signals = BackPropagatorSignals()
        self.dtype = np.complex64
        self._amplitude = amplitude
        self._phase = phase
        self.propagator = Propagator(amplitude, phase, pixelsize, wavelength, dtype=self.dtype, **kwargs)
        self.device = device
        self.z = z
        self.iter_num = iter_num
        self.amplitudes = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self.phases = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self._i = 0

    @QtCore.pyqtSlot()
    def run(self):

        if self.device is not None:
            self.propagator.init_gpu(cldevices=[self.device,])

        for i, zi in enumerate(self.z):
            for _ in range(self.iter_num):
                if self._i == 0:
                    self.propagator.backpropagate(zi)
                else:
                    self.propagator.apply_constrain_nonneg()
                    self.propagator.propagate(zi)
                    self.propagator.apply_constrain_hologram()
                    self.propagator.backpropagate(zi)
                self._i += 1
            self._i = 0
            field = self.propagator.field
            self.amplitudes[...,i] = np.abs(field)
            self.phases[...,i] = self.processPhases(np.angle(field))
            self.signals.progress.emit(int(100*(i+1)/self.z.size))

        self.signals.finished.emit(self.amplitudes, self.phases)

    def processPhases(self, phase):
        hist, bin_edges = np.histogram(phase, bins=int(np.sqrt(phase.size)))
        centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
        x0 = centers[np.argmax(hist)]

        y = phase - x0
        y[y > np.pi] -= 2*np.pi
        y[y < -np.pi] += 2*np.pi

        return y

class IteratorThread(QtCore.QThread):

    def __init__(self, amplitude, phase, pixelsize, wavelength, z, iter_num, receive_termination, device=None, **kwargs):
        super().__init__()
        self.signals = BackPropagatorSignals()
        self.dtype = np.complex64
        self._amplitude = amplitude
        self._phase = phase
        self.propagator = Propagator(amplitude, phase, pixelsize, wavelength, dtype=self.dtype, **kwargs)
        self.device = device
        self.z = z
        self.iter_num = iter_num
        self.amplitudes = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self.phases = np.zeros((*amplitude.shape, self.z.size), dtype=np.float32)
        self._i = 0
        self._terminate = False

        # connect to the termination check
        receive_termination.connect(self.termination_check)

    def run(self):

        if self.device is not None:
            self.propagator.init_gpu(cldevices=[self.device,], clbuild_options=['-cl-fast-relaxed-math', '-cl-mad-enable'])

        start = time.perf_counter()
        for i, zi in enumerate(self.z):
            for _ in range(self.iter_num):
                self.signals.check_termination.emit()
                if self._terminate:
                    print('Received the break signal.')
                    break
                if self._i == 0:
                    self.propagator.backpropagate(zi)
                else:
                    self.propagator.apply_constrain_nonneg()
                    self.propagator.propagate(zi)
                    self.propagator.apply_constrain_hologram()
                    self.propagator.backpropagate(zi)
                self._i += 1

            if self._terminate:
                break

            self._i = 0
            field = self.propagator.field
            self.amplitudes[...,i] = np.abs(field)
            self.phases[...,i] = self.processPhases(np.angle(field))

            self.propagator.reset()
            self.signals.progress.emit(int(100*(i+1)/self.z.size))

        print('Finished in:', time.perf_counter() - start,'s')
        self.signals.finished.emit(self.amplitudes, self.phases)

    def processPhases(self, phase):
        hist, bin_edges = np.histogram(phase, bins=int(np.sqrt(phase.size)))
        centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
        x0 = centers[np.argmax(hist)]

        y = phase - x0
        y[y > np.pi] -= 2*np.pi
        y[y < -np.pi] += 2*np.pi

        return y

    def termination_check(self, termination):
        self._terminate = termination

class SlicerWindow(SlicerTemplateBaseClass):

    def __init__(self, z: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray):
        SlicerTemplateBaseClass.__init__(self)
        self.ui = SlicerWindowTemplate()
        self.ui.setupUi(self)

        self.z = z
        self.amplitudes = amplitudes
        self.phases = phases

        self.viewbox1 = CustomViewBox(invertY=True, lockAspect=True)
        self.ui.gwAmplitude.setCentralWidget(self.viewbox1)
        self.imgitem1 = CustomImageItem()
        self.imgitem1.axisOrder = 'row-major'
        self.viewbox1.addItem(self.imgitem1)
        self.ui.gwAmpHist.setCentralWidget(CustomHistogramLUTItem(image=self.imgitem1))
        self.imgitem1.setImage(self.amplitudes[...,0])

        self.viewbox2 = CustomViewBox(invertY=True, lockAspect=True)
        self.ui.gwPhase.setCentralWidget(self.viewbox2)
        self.imgitem2 = CustomImageItem()
        self.imgitem2.axisOrder = 'row-major'
        self.viewbox2.addItem(self.imgitem2)
        self.ui.gwPhHist.setCentralWidget(CustomHistogramLUTItem(image=self.imgitem2))
        self.imgitem2.setImage(self.phases[...,0])

        # functionality with the data points
        self.imgitem1.sendDataSignal.connect(self.setCoorsValues)
        self.imgitem2.sendDataSignal.connect(self.setCoorsValues)

        # make a connection to the scrollbar and linetextedit
        self.ui.horizontalSliderReconst.setSingleStep(1)
        self.ui.horizontalSliderReconst.setMinimum(0)
        self.ui.horizontalSliderReconst.setMaximum(self.z.size-1)
        self.ui.horizontalSliderReconst.valueChanged.connect(self.updateSlicerSlider)
        self.ui.lineEditzReconst.returnPressed.connect(self.updateSlicerEnter)

        self.show()

    def setCoorsValues(self, data):
        self.ui.labelValue.setText('{:10.3f}'.format(data['data']))
        self.ui.labelXYcoor.setText('x ={:9.3f}, y ={:9.3f}'.format(data['pos'][0], data['pos'][1]))

    def updateSlicerSlider(self):

        index = self.ui.horizontalSliderReconst.value()
        z = self.z[index]
        self.ui.lineEditzReconst.setText('{:10.4f}'.format(1e6*z))
        self.imgitem1.setImage(self.amplitudes[...,index])
        self.imgitem2.setImage(self.phases[...,index])

    def updateSlicerEnter(self):
        z = 1e6 * float(self.ui.lineEditzReconst.text())  # check that the data entered is reasonable

# class SlicerWindow(SLicerTemplateBaseClass):
#     def __init__(self, propagator: Propagator, 
#         z:np.ndarray or tuple or list, reconstruct=False,
#             iterations=50):

#         """
#         Offers a view through slices obtained via angular spectrum method.

#         Parameters
#         ----------

#         propagator: Propagator
#             Instance of Propagator class holding the methods for
#             angular spectrum. If reconstruct=True, propagator should be 
#             initialized at hologram plane, if reconstruct=False, propagator
#             should be at initialized at object plane, i.e. provided from
#             the iterator.

#         z: nd.array or tuple or list
#             Array of z values at which the slices are computed.

#         reconstruct: bool
#             If True, the Slicer will iteratively reconstruct for each z.

#         iterations: int
#             Number of iterations in case reconstruct is True.

#         """
#         SLicerTemplateBaseClass.__init__(self)
        
#         # create the slicer window
#         self.ui = SlicerWindowTemplate()
#         self.ui.setupUi(self)
#         self.title = self.ui.label
#         self.slider = self.ui.horizontalSlider

#         self.graphicslayout = self.ui.graphicsView
#         self.viewBox_amplitude = self.graphicslayout.addViewBox(row=0 , col=0)
#         self.viewBox_phase = self.graphicslayout.addViewBox(row=0 , col=1)
#         self.viewBox_amplitude.invertY(True)
#         self.viewBox_amplitude.setAspectLocked(True)
#         self.viewBox_phase.invertY(True)
#         self.viewBox_phase.setAspectLocked(True)

#         # parameters
#         self.nz = len(z)
#         self.ind = self.nz//2
#         self.propagator = propagator
#         self.reconstruct = reconstruct
#         self.z = np.sort(np.array(z))

#         if reconstruct:
#             self.title.setText(self.title.text().replace('placeholder_1', 'Reconstruction'))
#         else:
#             self.stepz = np.append(self.z[0], self.z[1:]-self.z[:-1])
#             self.title.setText(self.title.text().replace('placeholder_1', 'Propagation'))

#         self.title.setText(self.title.text().replace(
#             'placeholder_2', '{:.4f}'.format(1e6*self.z[self.ind])))
#         self.field = np.zeros((*propagator.field.shape, self.nz), dtype=np.complex)

#         # do the propagation or reconstruction
#         if reconstruct:
#             for i, z_i in enumerate(self.z):
#                 it = Iterator(z_i, copy.deepcopy(propagator))
#                 it.iterate(iterations)
#                 self.field[..., i] = it.propagator.field
#             print('Finished reconstruction!')
#         else:
#             for i, stepz_i in enumerate(self.stepz):
#                 propagator.propagate(stepz_i)
#                 self.field[..., i] = propagator.field
#             print('Finished propagation!')

#         self.imgitem_amplitude = pg.ImageItem(np.abs(self.field[..., self.ind]).T)
#         self.viewBox_amplitude.addItem(self.imgitem_amplitude)
#         self.imgitem_phase = pg.ImageItem(np.angle(self.field[..., self.ind]).T)
#         self.viewBox_phase.addItem(self.imgitem_phase)

#         # set the functionality of the slider
#         self.slider.setSliderPosition(self.ind)
#         self.slider.setSingleStep(1)
#         self.slider.setMinimum(0)
#         self.slider.setMaximum(self.nz - 1)
#         self.slider.sliderMoved.connect(self.update)

#     def update(self, index):
#         print(index)
#         self.imgitem_amplitude.setImage(np.abs(self.field[..., index]).T)
#         self.imgitem_phase.setImage(np.angle(self.field[..., index]).T)

if __name__ == '__main__':

    # PREPROCESSING
    # input files
    # hologram_file = 'foc25.npz'
    # background_file = 'nosamp.npz'
    # transmission_glass = 0.92

    # load the files and subtract background
    # data = np.load(hologram_file)
    # data_bg = np.load(background_file)
    # hologram = data['data']/(transmission_glass * data_bg['data'])

    # APPLICATION
    app = QtWidgets.QApplication([])
    window = MainWindow()
    app.exec()