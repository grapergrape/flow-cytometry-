# -*- coding: utf-8 -*-
import sys
import time
import os.path
import queue
import traceback

import threading
import multiprocessing

import numpy as np
import scipy.io

from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog
from PyQt5.QtGui import QPen
from PyQt5.QtCore import QTimer, QCoreApplication, Qt

from widgets import resources, common, dialogs, imview, plot
from widgets.common import prepareqt
from widgets.autoexposure import AutoExposureDialog as AutoExposure
from common.acquisition import Acquisition

DEFAULT_ICON_WIDTH = 48

ACQ_WHITE_DARK = {
    'label':('Turn on the light source and start acquisition when ready',
             'Turn off the light source and start acquisition when ready',),
    'icon':('lightsource_on.png', 'lightsource_off.png',)
}


class AcquireImages(common.QBackgroundTask):

    def __init__(self, parent=None):
        '''
        Acquires images in a background thread and shows a progress window
        that allows user to interrupt the process.

        Use the start method to start the acquisition in a background thread.

        Once complete, the acquired images in a form of Acquisition object
        can be retrieved by calling the data method.
        '''
        common.QBackgroundTask.__init__(self, parent=parent)

    def start(self, camera: object, acquisition: Acquisition = None,
              **kwargs) -> bool:
        '''
        Starts the acquisition in a background process.

        Parameters
        ----------

        camera: device.Device
            Camera device.

        acquisition: Acquisition
            Acquisition object passed to the camera.acquire method. If None,
            a new acquisition object is constructed using the provided
            keyword arguments (kwargs) as Acquisition(**kwargs).

        kwargs: dict
            Arguments passed to the Acquisition constructor if the value of
            parameter acquisition id None.

        Once complete, the acquired images in a form of Acquisition object
        can be retrieved by calling the result method.

        Returns
        -------
        result: bool
            True on success, False otherwise.
        '''

        if acquisition is None:
            try:
                acquisition = Acquisition(**kwargs)
            except:
                details = traceback.format_exc()
                self._reportError(
                    QCoreApplication.translate(
                        'AcquireImages',
                        'Failed to create an Acquisition instance!'
                    ),
                    details)
                self._finalize()
                return

        return common.QBackgroundTask.start(
            self,
            target=self.task,
            kwargs={'camera':camera, 'acquisition':acquisition},
            title=QCoreApplication.translate(
                'AcquireImages',
                'Acquiring data'
            ),
            label=QCoreApplication.translate(
                'AcquireImages',
                'Acquiring the requested data.'
            )
        )

    def task(self, camera, acquisition):
        result = None
        try:
            tickTime = acquisitionTime = None
            if camera.isProperty('framerate'):
                t = camera.get('exposuretime')
                framerate = camera.get('framerate')
                frameTime = max(t, 1.0/framerate)
                acquisitionTime = acquisition.get('n')*frameTime*\
                    camera.get('averaging')
                tickTime = acquisitionTime/100.0

            pollPeriod = 0.1

            camera.acquire(acquisition)
            if tickTime is not None:
                startTime = time.perf_counter()
            while 1:
                if tickTime is not None:
                    delta = time.perf_counter() - startTime
                    currentTicks = min(int(delta/tickTime + 0.5), 99)
                else:
                    currentTicks = min(int(acquisition.get('acquired')/
                                           acquisition.get('n')*100 + 0.5), 99)
                # set the progress bar value
                self.setProgress(currentTicks)
                # check for pending stop request
                if self.wasCanceled():
                    camera.stop()
                    break
                if acquisition.join(pollPeriod):
                    result = acquisition
                    break
        except Exception:
            details = traceback.format_exc()
            self.reportError(
                QCoreApplication.translate(
                    'AcquireImages',
                    'Unhandled exception occurred during '\
                    'acquisition of data!'
                ),
                details
            )

        return result

#%% Base class for creating previews in an independent process
class RemotePreview:
    MAX_FPS = 20.0
    MIN_FPS = 0.1
    TIMEOUT = 5

    _PREVIEWS = []
    _PREVIEW_RLOCK = threading.RLock()

    @staticmethod
    def _updateFromDevice(targetobj, device, fps, **kwargs):
        fps = min(max(fps, RemotePreview.MIN_FPS), RemotePreview.MAX_FPS)
        while targetobj.isAlive():
            # do not forward None data
            im = targetobj.prepareData(device, **kwargs)
            if im is not None:
                targetobj.sendData(im)
            time.sleep(1.0/fps)

    @staticmethod
    def open(targeobj, device, fps=10, **kwargs):
        previewThread = threading.Thread(
            target=RemotePreview._updateFromDevice,
            args=(targeobj, device, fps), kwargs=kwargs)
        previewThread.start()
        return targeobj

    @staticmethod
    def closeAll():
        with RemotePreview._PREVIEW_RLOCK:
            nitems = len(RemotePreview._PREVIEWS)
            for i in range(nitems):
                RemotePreview._PREVIEWS.pop().close()

    #%% Client side API
    def __init__(self, title=None, **kwargs):
        self._iqueue = multiprocessing.Queue(1)
        self._proc = multiprocessing.Process(
            target=self._targetProc,
            args=(self._iqueue, title,), kwargs=kwargs)
        self._proc.start()

        self._apirlock = threading.RLock()
        with RemotePreview._PREVIEW_RLOCK:
            RemotePreview._PREVIEWS.append(self)

        self._qtapp = self._updateTimer = self._iqueu = self._targetObj = None

    def sendData(self, data):
        with self._apirlock:
            try:
                if self.isAlive():
                    self._iqueue.put_nowait(data)
            except queue.Full:
                pass

    def close(self):
        with self._apirlock:
            if self.isAlive():
                try:
                    self._iqueue.put(None, False, RemotePreview.TIMEOUT)
                    #self._iqueue.close()
                except queue.Full:
                    self._proc.terminate()
                self._iqueue.close()
                self._proc.join(RemotePreview.TIMEOUT)

        with RemotePreview._PREVIEW_RLOCK:
            if self in RemotePreview._PREVIEWS:
                RemotePreview._PREVIEWS.remove(self)

    def isAlive(self):
        return self._proc.is_alive()

    def join(self, timeout=None):
        self._proc.join(timeout)
        return self._proc.is_alive()

    def prepareData(self, device, **kwargs):
        # overload and prepare data object to be sent to the viewer here
        return None

    #%% Viewer rocess side API

    def _targetProc(self, iqueue, title, **kwargs):
        self._iqueu = iqueue
        self._targetObj = self._updateTimer = None
        try:
            self._qtapp = prepareqt()

            if title is None:
                title = QCoreApplication.translate('_targetProc', 'Preview')

            self._targetObj = self.initTarget(title, **kwargs)

            self._updateTimer = QTimer()
            self._updateTimer.timeout.connect(self._updateTarget)
            self._updateTimer.start(int(1000//RemotePreview.MAX_FPS))

            self._qtapp.exec_()
        finally:
            if self._updateTimer is not None:
                self._updateTimer.stop()

        self._iqueu = None
        iqueue.close()
        iqueue.join_thread()

        return 0

    def _updateTarget(self):
        try:
            data = self._iqueue.get_nowait()
            if data is None:
                self._qtapp.quit()
            else:
                self.updateTarget(self._targetObj, data)
        except queue.Empty:
            pass
        except:
            # unexpected error .. terminate application
            self._qtapp.quit()

    def initTarget(self, title, **kwargs):
        # overload and initialize/create the viewer/target here and return it
        return None

    def updateTarget(self, target, data):
        # overload and define custom viewer/target update
        pass

#%% Camera preview class
class CameraPreview(RemotePreview):
    @staticmethod
    def open(camera, title=None, fps=10):
        # start preview if required
        if not camera.acquiring():
            camera.acquire()
        viewobj = CameraPreview(title, span=(0, 2**camera.get('bitdepth') - 1))
        return RemotePreview.open(viewobj, camera, fps)

    def __init__(self, title=None, span=(0, 2**14 - 1,), **kwargs):
        RemotePreview.__init__(self, title, span=span, **kwargs)

    def prepareData(self, camera):
        return camera.get('lastframe'), camera.get('acquired')

    def initTarget(self, title, **kwargs):
        span = kwargs['span']

        view = imview.Image(levels=span)
        view.setRange(span)
        view.setLevels(span)
        view.hideProfiles()
        view.setWindowTitle(title)
        view.setMinimumSize(800, 600)

        view.show()

        return view

    def updateTarget(self, view, data):
        image, nacquired = data
        view.setImage(image, info='Frame {}'.format(nacquired))

#%% Spectrometer preview class
class SpectrometerPreview(RemotePreview):
    @staticmethod
    def open(spectrometer, title=None, fps=10):
        # start preview if required
        if not spectrometer.acquiring():
            spectrometer.acquire()
        viewobj = SpectrometerPreview(
            title, span=(0, 2**spectrometer.get('bitdepth') - 1))
        return RemotePreview.open(viewobj, spectrometer, fps)

    def __init__(self, title=None, span=(0, 2**14 - 1,), **kwargs):
        RemotePreview.__init__(self, title, span=span, **kwargs)

    def prepareData(self, spectrometer):
        spectrum = spectrometer.get('newspectrum')
        if spectrum is not None:
            return spectrometer.get('wavelength'), spectrum, \
                   spectrometer.get('acquired')

    def initTarget(self, title, **kwargs):
        span = kwargs['span']

        view = plot.Plotx()
        view.setGridVisible(True)
        view.setYRange(*span)
        view.maxIntensityHLineItem = view.hLine(span[1], pen=QPen(Qt.red, 0))
        view.previewPlotItem = view.plot([], [], pen=QPen(Qt.blue, 0))
        view.hideCursorLine()
        view.setWindowTitle(title)
        view.lockedPushButton = common.QLockUnlockPushButton()
        view.lockedPushButton.setToolTip(
            QCoreApplication.translate(
                'SpectrometerPreview', 'Lock the current spectrum.')
        )
        view.exportPushButton = common.QSquarePushButton(
            resources.loadIcon('export.png'), '')
        view.exportPushButton.setToolTip(
            QCoreApplication.translate(
                'SpectrometerPreview', 'Save the current spectrum to a file.')
        )
        view.exportPushButton.clicked.connect(
            lambda state: self.exportSpectrum(view))
        view.statusBar().layout().addWidget(view.lockedPushButton)
        view.statusBar().layout().addWidget(view.exportPushButton)
        view.setMinimumSize(800, 600)
        view.show()

        return view

    def exportSpectrum(self, view, filename=None):
        wavelengths = view.previewPlotItem.xData
        spectrum = view.previewPlotItem.yData
        if spectrum is not None and spectrum.size <= 0:
            dialogs.InformationMessage(
                view,
                QCoreApplication.translate(
                    'SpectrometerPreview',
                    'Save spectrum to a file'),
                QCoreApplication.translate(
                    'SpectrometerPreview',
                    'Nothing to save.'),
            ).exec()
            return
        if filename is None:
            filename = QFileDialog.getSaveFileName(
                view,
                QCoreApplication.translate(
                    'SpectrometerPreview',
                    'Save spectrum to a file'),
                '',
                QCoreApplication.translate(
                    'SpectrometerPreview',
                    'Numpy data file *.npz;;Matlab data file *.mat'
                )
            )
            if isinstance(filename, tuple):
                filename = filename[0]
        if filename:
            try:
                target = os.path.splitext(filename)[1].lower()
                data = {'spectrum':spectrum, 'wavelengths':wavelengths}
                if target == '.npz':
                    np.savez_compressed(filename, **data)
                elif target == '.mat':
                    scipy.io.savemat(filename, data)
                else:
                    dialogs.ErrorMessage(
                        view,
                        QCoreApplication.translate(
                            'SpectrometerPreview',
                            'Save calibration package'
                        ),
                        QCoreApplication.translate(
                            'SpectrometerPreview',
                            'The specified file format ({}) is not supported!'
                        ).format(target),
                    )
            except:
                dialogs.ErrorMessage(
                    view,
                    QCoreApplication.translate(
                        'SpectrometerPreview',
                        'Save spectrum error'),
                    QCoreApplication.translate(
                        'SpectrometerPreview',
                        'Failed to save the spectrum '\
                        'to the specified file!'),
                    traceback.format_exc(),
                ).exec()

    def updateTarget(self, view, data):
        if not view.lockedPushButton.isChecked():
            wavelengths, spectrum, nacquired = data
            view.setPlotInfo('Spectrum {}'.format(nacquired))
            view.previewPlotItem.setData(wavelengths, spectrum)
#%% Auto exposure
def autoExposure(device, intensity=0.85, parent=None, **kwargs):
    dialog = AutoExposure(parent)

    if dialog.start(device, intensity, **kwargs):
        device.set('exposuretime', dialog.result())
        return device.get('exposuretime')
    return None

#%% Acquisition of images and spectra
def _acquire(device, acq, averaging, doautoexposure, parent=None, **kwargs):

    # save the camera settings
    exposure_time_state = device.get('exposuretime')
    averaging_state = device.get('averaging')
    acquiring_state = device.acquiring()

    # auto adjust exposure time if required
    if doautoexposure and autoExposure(device, parent=parent, **kwargs) is None:
        acq = None

    if acq is not None:
        device.stop()
        device.set('averaging', averaging)
        acqdlg = AcquireImages(parent)
        if not acqdlg.start(device, acq):
            acq = None

    # restore camera settings
    device.stop()
    device.set('averaging', averaging_state)
    device.set('exposuretime', exposure_time_state)
    if acquiring_state:
        device.acquire()

    return acq

def acquireImages(camera, n, averaging=1, doautoexposure=False,
                  startdialog=False, title=None, label=None,
                  parent=None, iconwidth=None, acqclass=None, **kwargs):

    if n >= 1:

        if iconwidth is None:
            iconwidth = DEFAULT_ICON_WIDTH

        if parent is None:
            prepareqt()

        if kwargs is None:
            kwargs = {}

        ready = True
        if startdialog:
            if title is None:
                title = QCoreApplication.translate(
                    'acquireImages', 'Acquire images')
            if label is None:
                label = QCoreApplication.translate(
                    'acquireImages', 'Start acquisition when ready.')
            dialog = dialogs.QuestionMessage(parent, title, label)
            dialog.setIconPixmap(
                resources.loadPixmap('camera.png').scaledToWidth(iconwidth))
            dialog.button(QMessageBox.Yes).setText(
                QCoreApplication.translate('acquireImages', 'Continue')
            )
            dialog.button(QMessageBox.No).setText(
                QCoreApplication.translate('acquireImages', 'Cancel')
            )

            ready = dialog.exec() == QMessageBox.Yes

        if ready:
            if acqclass is None:
                acqclass = Acquisition
            acq = acqclass(n)
            return _acquire(camera, acq, averaging, doautoexposure, parent,
                            **kwargs)

    return None

def acquireImagePair(camera, n, averaging=1, doautoexposure=False,
                     title=None, cfg=None, parent=None,
                     iconwidth=None, acqclass=None, **kwargs):

    if cfg is None:
        cfg = ACQ_WHITE_DARK

    if iconwidth is None:
        iconwidth = DEFAULT_ICON_WIDTH

    if parent is None:
        prepareqt()

    if kwargs is None:
        kwargs = {}


    if n >= 1:
        if title is None:
            title = QCoreApplication.translate(
                'acquireImages', 'Acquire a pair of images')

        dialog = dialogs.QuestionMessage(None, title, cfg['label'][0])
        dialog.setIconPixmap(
            resources.loadPixmap(cfg['icon'][0]).scaledToWidth(iconwidth))
        dialog.button(QMessageBox.Yes).setText(
            QCoreApplication.translate('acquireImagePair', 'Continue')
        )
        dialog.button(QMessageBox.No).setText(
            QCoreApplication.translate('acquireImagePair', 'Cancel')
        )

        if dialog.exec() == QMessageBox.Yes:
            if acqclass is None:
                acqclass = Acquisition

            acq_1 = acqclass(n)
            acq_1 = _acquire(camera, acq_1, averaging, doautoexposure, parent,
                             **kwargs)

            if acq_1 is not None:
                dialog.setIconPixmap(
                    resources.loadPixmap(
                        cfg['icon'][1]).scaledToWidth(iconwidth))
                dialog.setText(cfg['label'][1])
                if dialog.exec() == QMessageBox.Yes:
                    acq_2 = acqclass(n)
                    acq_2 = _acquire(camera, acq_2, averaging,
                                     doautoexposure, parent, **kwargs)
                if acq_2 is not None:
                    return acq_1, acq_2

    return None
