# -*- coding: utf-8 -*-
"""
Created on Wen Feb 20 22:51:01 2017

@author: miran
"""
import os.path
import traceback
import time
import copy

import numpy as np
import scipy.io

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, \
                            QHBoxLayout, QSplitter, QComboBox, QListView, \
                            QDoubleSpinBox, QDialog, QTabWidget, QTabBar
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QCoreApplication, QTimer, QSize
from PyQt5.QtCore import pyqtSignal as QSignal

PICKLE_PROTOCOL = 3

from widgets import plot, common, resources, autoexposure
from common import misc

DEFAULT_CONFIGURATION = {
    'ui':{
        # preview refresh period (s)
        'refreshperiod': 0.1,

        # acquisition averaging
        'averagingrange': (1, 128),
        # initial value of the acquisition averaging
        'averaging': 1,

        # exposure time range, use None for camera default
        'exposuretimerange': None,
        # number of steps available at the exposure time slider
        'exposuretimesteps': 200,
        # number of decimals used to display the slider position (None for auto)
        'exposuretimedecimals': 6,

        # gain range, use None for camera default
        'gainrange': None,
        # number of decimals used to display the slider position (None for auto)
        'gaindecimals': None,
        # step of the gain slider, set to None for default
        'gainstep': None,
        # explicitly disable offset adjustment
        'disable_gain': False,

        # black level offset range, use None for camera default
        'offsetrange': None,
        # number of steps available at the black level offset slider
        'offsetdecimals': None,
        # number of decimals used to display the slider position (None for auto)
        'offsetstep': None,
        # explicitly disable offset adjustment
        'disable_offset': False,

        # explicitly disable horizontal flipping of image
        'disable_flip_horizontally': False,

        # explicitly disable vertical flipping of image
        'disable_flip_vertically': False,

        # autoexposure configuration
        'autoexposure':{
            'statistics': 'max',
            'target': 85.0,
            'targetrange': (0.0, 100.0),
            'percentile': 85.0,
            'percentilerange': (5.0, 100.0),
            'verbose': False
        },

        # Pause - resume device acquisition when accessing device properties
        #   such as gain, offset, temperature, ...
        'safedeviceaccess': False,
    },
}

def clip(value, low, high):
    return max(min(value, high), low)


class DeviceGuard:
    def __init__(self, dev):
        self._device = dev
        self._acquiring = False
        self._enabled = True

    def __enter__(self):
        if self._enabled:
            self._acquiring = self._device.acquiring()
            self._device.pause()

    def __exit__(self, type, value, traceback):
        #Exception handling here
        if self._enabled:
            if self._acquiring:
                self._device.resume()

    def setEnabled(self, state):
        self._enabled = bool(state)

    def enabled(self):
        return self._enabled


class Hub(QWidget):
    def __init__(self, *args, **kwargs):
        '''
        Device Hub base class. Use this class to derive a custom device hub.

        Parameters
        ----------
        args, kwargs: list, dict:
            Arguments passed to the QWidget superclass constructor.
        '''
        super().__init__(*args, **kwargs)

        self._closableViewTabs = []

        # Tab widget that will hold views of hub modules
        self.viewTabWidget = QTabWidget()
        self.viewTabWidget.setTabsClosable(True)

        # Tab widget that will hold the configuration/control of hub modules
        self.controlTabWidget = QTabWidget()
        self.controlTabWidget.setTabsClosable(False)
        self.controlTabWidget.tabCloseRequested.connect(lambda index: None)

        self._hsplitter = QSplitter(Qt.Horizontal)
        self._hsplitter.addWidget(self.viewTabWidget)
        self._hsplitter.addWidget(self.controlTabWidget)
        self._hsplitter.setStretchFactor(0, 1)
        self._hsplitter.setStretchFactor(1, 0)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._hsplitter)
        self.setLayout(layout)

        # enable synchronized switching of tabs
        self.controlTabWidget.currentChanged.connect(self.selectControlTab)

        self._viewTabIndices = {}
        self._controTabIndices = {}

        self._devices = {}

    def selectControlTab(self, module: int or QWidget):
        '''
        Select/activate control tab for the given module.

        Parameters
        ----------
        module: HubModule or int
            Module instance or tab index that contains the module.
        '''
        if isinstance(module, int):
            for key, value in self._controTabIndices.items():
                if value == module:
                    module = key
                    break

        if module in self._controTabIndices:
            self.controlTabWidget.setCurrentIndex(
                self._controTabIndices[module])
            view_index = self._viewTabIndices.get(module)
            if view_index is not None:
                self.viewTabWidget.setCurrentIndex(view_index)

    def modules(self) -> list:
        '''
        A list of modules connected to this hub

        Returns
        -------
        modules: list
            A list of modules connected to this hub.
        '''
        return list(self._controTabIndices.keys())

    def device(self, deviceid: str) -> object:
        '''
        Get the specified hardware device instance from the hub.

        Parameters
        ----------
        deviceid: str
            Device ID that was used to add the device to the hub by the
            addDevice method.

        Returns
        -------
        device: object
            Device instance. Raises ValueError if a device with the given ID is
            not found in the hub.
        '''
        return self._devices.get(deviceid)

    def addDevice(self, deviceid: str, device: object):
        '''
        Add hardware device with the specified identifier.

        Parameters
        ----------
        deviceid: str
            Device identifier.
        device: object
            Hardware device to add to this hub. Raises ValueError if the
            given deviceid is already in use.
        '''
        if deviceid in self._devices:
            raise ValueError(
                'Device "{}" is already in use by the hub!'.format(deviceid)
            )
        self._devices[deviceid] = device

    def connectModule(self, module, closable: bool = False):
        '''
        Connect a hub module to this hub.

        Parameters
        ----------
        module: HubModule
            Module instance.
        closable: bool
            Weather the tab that holds the view widget of the module
            should be closable.
        '''
        if module not in self._controTabIndices:
            # prepare module for this hub
            module.connectHub(self)

            self._controTabIndices[module] = self._addModuleControl(
                module, label=module.label(), icon=module.icon(),
                tooltip=module.tooltip()
            )
            view = module.view()
            if view is not None:
                self._viewTabIndices[module] = self._addModuleView(
                    view, label=module.label(), icon=module.icon(),
                    tooltip=module.tooltip(), closable=closable
                )

    def disconnectModule(self, module):
        '''
        Disconnect module from the hub.
        '''
        if module in self._controTabIndices:
            module.disconnectHub(self)

            self.controlTabWidget.removeTab(self._controTabIndices[module])
            if module in self._viewTabIndices:
                self.viewTabWidget.removeTab(self._viewTabIndices[module])
                self._viewTabIndices.pop(module)

            self._controTabIndices.pop(module)

    def _addModuleView(self, viewWidget: QWidget, label: str,
                       icon: QIcon = None, tooltip: str = None,
                       closable: bool = False):
        '''
        Add view widget of a the hub module (returned by the view method
        of the module).

        Parameters
        ----------
        viewWidget: QWidget
            View widget instance.
        label: str
            Label used as the tab text.
        icon: QIcon
            Icon used as the tab icon.
        tooltip: str
            Optional tab tooltip.
        closable: bool
            A flag indicating weather the tab with the given view is closable.
        '''
        tab_index = self.viewTabWidget.addTab(viewWidget, label)
        if icon is not None:
            self.viewTabWidget.setTabIcon(tab_index, icon)
        if tooltip is not None:
            self.viewTabWidget.setTabToolTip(tab_index, tooltip)
        if closable:
            self._closableViewTabs.append(tab_index)
        else:
            self.viewTabWidget.tabBar().\
                tabButton(tab_index, QTabBar.RightSide).setVisible(False)
        return tab_index

    def _addModuleControl(self, controlWidget: QWidget, label: str,
                          icon: QIcon = None, tooltip: str = None):
        '''
        Add control widget of a the hub module (the hub module itself).

        Parameters
        ----------
        controlWidget: QWidget
            Control widget instance.
        label: str
            Label used as the tab text.
        icon: QIcon
            Icon used as the tab icon.
        tooltip: str
            Optional tab tooltip.
        '''
        tab_index = self.controlTabWidget.addTab(controlWidget, label)
        if icon is not None:
            self.controlTabWidget.setTabIcon(tab_index, icon)
        if tooltip is not None:
            self.controlTabWidget.setTabToolTip(tab_index, tooltip)
        return tab_index

    def closeEvent(self, event):
        '''
        Handling hub window close events. Default implementation stops
        acquisition on all hub devices.
        '''
        for device_id, device in self._devices.items():
            if hasattr(device, 'stop'):
                device.stop()

    def sizeHint(self) -> QSize:
        '''
        Initial hub window size hint.
        '''
        return QSize(1024, 600)


class HubModule(QWidget):
    def __init__(self, *args, **kwargs):
        '''
        Hub module base class. Use this class to derive a custom hub module.
        This class serves as the hub control widget. The view method should
        return a view widget associated with the hub module.

        Parameters
        ----------
        args, kwargs: list, dict:
            Arguments passed to the QWidget superclass constructor.
        '''
        super().__init__(*args, **kwargs)
        self._hub = None

    def label(self) -> str:
        '''
        Returns default hub module label text used in the tab.
        '''
        return QCoreApplication.translate(
            'HubModule', 'Module'
        )

    def tooltip(self) -> str:
        '''
        Returns default hub module tooltip text used in the tab.
        '''
        return QCoreApplication.translate(
            'HubModule', 'Module'
        )

    def description(self) -> str:
        '''
        Returns default hub module description.
        '''
        return QCoreApplication.translate(
            'HubModule', 'Module description'
        )

    def icon(self):
        '''
        Returns default hub module icon.
        '''
        return resources.loadIcon('empty.png')

    def view(self):
        '''
        Returns a view widget associated with the hub module.
        '''
        return None

    def connectHub(self, hub: Hub):
        '''
        Connect module to the hub. Called by the hub.
        Do not explicitly call this method, except in the overloaded method
        of a sublass.
        '''
        self._hub = Hub

    def disconnectHub(self):
        '''
        Disconnect module from the hub. Called by the hub.
        Do not explicitly call this method, except in the overloaded method
        of a sublass.
        '''
        self._hub = None

    def hub(self) -> Hub:
        '''
        Get the module hub.
        '''
        return self._hub


class FpaHubModule(HubModule):
    exposureTimeChanged = QSignal(object)
    gainChanged = QSignal(object)
    offsetChanged = QSignal(object)
    averagingChanged = QSignal(object)
    newPreviewItem = QSignal(tuple)

    # Timw window (seconds) used for software estimation of framerate if
    # not available through the hardware
    FRAMERATE_ESTIMATION_WINDOW = 5

    def __init__(self, device, cfg=None, **kwargs):
        '''
        Base class for Focal Plane Array (FPA) control modules.

        Parameters
        ----------
        device: device.Device
            Hardware device instance.
        cfg: dict
            Hub module configuration. See DEFAULT_CONFIGURATION for more
            details.
        kwargs: dict
            Arguments passed to the superclass HubModule constructor.
        '''
        super().__init__(**kwargs)

        if cfg is None:
            cfg = DEFAULT_CONFIGURATION
        else:
            cfg = misc.mergeTwoDictsDeep(DEFAULT_CONFIGURATION, cfg)
        self._cfg = cfg
        uicfg = cfg['ui']

        self._cfg = cfg
        self._device = device
        self._previewItemIndex = 0
        self._previousTimestamp = self._previousAcquired = None
        self._framerateHistory = np.tile(-np.inf, (100, 2))

        self._deviceGuard = DeviceGuard(self._device)
        self._deviceGuard.setEnabled(uicfg['safedeviceaccess'])

        # prepare some default UI components for the hub
        self._label = QCoreApplication.translate(
            'FpaHubModule',
            'Focal plane array',
        )
        self._tooltip = QCoreApplication.translate(
            'FpaHubModule',
            'Focal plane array control module',
        )
        self._icon = resources.loadIcon('camera.png')

        self._description = QCoreApplication.translate(
            'FpaHubModule',
            'Focal plane array control module',
        )

        # exposure time control
        self.exposureTimeAdjustWidget = common.QAdjustWidget()
        et_range = uicfg.get('exposuretimerange')
        device_et_range = device.property('exposuretime').range
        if et_range is None:
            et_range = device_et_range
        et_range = (clip(et_range[0], *device_et_range),
                    clip(et_range[1], *device_et_range))
        steps = int(uicfg['exposuretimesteps'])
        ndecimals = uicfg['exposuretimedecimals']
        if et_range[0] != 0:
            self.exposureTimeAdjustWidget.setTransform(
                common.QAdjustWidget.FpLogTransform(
                    et_range[0], et_range[1], steps, ndecimals)
            )
        else:
            step = (et_range[1] - et_range[0])/steps
            self.exposureTimeAdjustWidget.setTransform(
                common.QAdjustWidget.FpTransform(
                    et_range[0], et_range[1], step)
            )
        self.exposureTimeAdjustWidget.setValue(
            self._device.get('exposuretime'))
        self.exposureTimeAdjustWidget.setGotoEndsVisible(True)
        self.exposureTimeAdjustWidget.setUnits('s')
        self.exposureTimeAdjustWidget.setCaption(
            QCoreApplication.translate(
                'FpaHubModule',
                'Exposure time'
            )
        )
        self.exposureTimeAdjustWidget.valueChanged.connect(
            self._updateExposureTime)

        # adjust gain
        self._layoutGain = self.gainCheckedWidget = self.gainAdjustWidget = None
        if self._device.isProperty('gain') and \
                self._device.property('gain').access.can('rw'):
            if self._device.property('gain').type in (float, int):
                self.gainAdjustWidget = common.QAdjustWidget()
                gain_range = uicfg.get('gainrange')
                spectrometer_gain_range = self._device.property('gain').range
                gain_type = self._device.property('gain').type
                if gain_range is None:
                    gain_range = spectrometer_gain_range
                gain_range = (clip(gain_range[0], *spectrometer_gain_range),
                              clip(gain_range[1], *spectrometer_gain_range))
                step = uicfg.get('gainstep')
                ndecimals = uicfg['gaindecimals']
                if gain_type == float:
                    if step is None:
                        step = 0.1
                    self.gainAdjustWidget.setTransform(
                        common.QAdjustWidget.FpTransform(
                            gain_range[0], gain_range[1], step, ndecimals)
                    )
                    #spectrometer.set('gain', float(gain_range[0]))
                else:
                    if step is None:
                        step = 1
                    self.gainAdjustWidget.setRange(
                        int(gain_range[0]), int(gain_range[1]))
                    self.gainAdjustWidget.setStep(step)
                    #spectrometer.set('gain', int(gain_range[0]))

                self.gainAdjustWidget.setValue(self._device.get('gain'))
                self.gainAdjustWidget.valueChanged.connect(self._updateGain)
                self.gainAdjustWidget.setGotoEndsVisible(True)
                self.gainAdjustWidget.setCaption(
                    QCoreApplication.translate(
                        'FpaHubModule',
                        'Gain'
                    )
                )
                self.gainAdjustWidget.valueChanged.connect(self._updateGain)
                if uicfg.get('disable_gain'):
                    self.gainAdjustWidget.setEnabled(False)
            elif self._device.property('gain').type is bool:
                self._device.set('gain', False)
                self.gainCheckedWidget = common.QCheckablePushButton()
                self.gainCheckedWidgetLabel = QLabel(
                    QCoreApplication.translate(
                        'FpaHubModule',
                        'High gain mode'
                    )
                )
                self._layoutGain = QHBoxLayout()
                self._layoutGain.addWidget(self.gainCheckedWidget)
                self._layoutGain.addWidget(self.gainCheckedWidgetLabel)
                self._layoutGain.addStretch()
                self.gainCheckedWidget.toggled.connect(self._updateGain)
                if uicfg.get('disable_gain'):
                    self.gainCheckedWidget.setEnabled(False)
                    self.gainCheckedWidgetLabel.setEnabled(False)

        # offset adjustment
        self.offsetAdjustWidget = None
        if self._device.isProperty('offset') and \
                self._device.property('offset').access.can('rw'):
            self.offsetAdjustWidget = common.QAdjustWidget()
            offset_range = uicfg.get('offsetrange')
            spectrometer_offset_range = self._device.property('offset').range
            offset_type = self._device.property('offset').type
            if offset_range is None:
                offset_range = spectrometer_offset_range
            offset_range = (clip(offset_range[0], *spectrometer_offset_range),
                            clip(offset_range[1], *spectrometer_offset_range))
            step = uicfg.get('offsetstep')
            ndecimals = uicfg['offsetdecimals']
            if offset_type == float:
                if step is None:
                    step = min((offset_range[1] - offset_range[0])/100.0, 1.0)
                self.offsetAdjustWidget.setTransform(
                    common.QAdjustWidget.FpTransform(
                        offset_range[0], offset_range[1], step, ndecimals)
                )
            else:
                if step is None:
                    step = 1
                self.offsetAdjustWidget.setRange(
                    int(offset_range[0]), int(offset_range[1]))
                self.offsetAdjustWidget.setStep(step)
            self.offsetAdjustWidget.setValue(self._device.get('offset'))
            self.offsetAdjustWidget.setGotoEndsVisible(True)
            self.offsetAdjustWidget.setCaption(
                QCoreApplication.translate(
                    'FpaHubModule',
                    'Black level offset'
                )
            )
            self.offsetAdjustWidget.valueChanged.connect(self._updateOffset)
            if uicfg.get('disable_offset'):
                self.offsetAdjustWidget.setEnabled(False)

        # averaging adjustment
        self.averagingAdjustWidget = None
        if self._device.isProperty('averaging') and \
                self._device.property('averaging').access.can('rw'):
            self.averagingAdjustWidget = common.QAdjustWidget()
            avg_range = uicfg.get('averagingrange')
            if avg_range is None:
                avg_range = self._device.property('averaging').range
            averaging = uicfg.get('averaging', 1)
            self.averagingAdjustWidget.setRange(
                int(avg_range[0]), int(avg_range[1]))
            self.averagingAdjustWidget.setValue(averaging)
            self.averagingAdjustWidget.setGotoEndsVisible(True)
            self.averagingAdjustWidget.setCaption(
                QCoreApplication.translate(
                    'FpaHubModule',
                    'Averaging'
                )
            )
            self.averagingAdjustWidget.valueChanged.connect(
                self._updateAveraging)

        # flip image/data horizontally
        layoutFlipHorizontally = None
        self.flipHorizontaliyCheckButton = self.flipHorizontallyLabel = None
        if self._device.isProperty('flipx') or \
                not uicfg.get('disable_flip_horizontally'):
            value = False
            if self._device.isProperty('flipx'):
                value = self._device.get('flipx')
            self.flipHorizontaliyCheckButton = common.QCheckablePushButton(
                value)
            self.flipHorizontaliyCheckButton.toggled.connect(
                self._updateFlipHorizontally)
            self.flipHorizontallyLabel = QLabel(
                QCoreApplication.translate(
                    'FpaHubModule',
                    'Flip image horizontally'
                )
            )
            layoutFlipHorizontally = QHBoxLayout()
            layoutFlipHorizontally.addWidget(self.flipHorizontaliyCheckButton)
            layoutFlipHorizontally.addWidget(self.flipHorizontallyLabel)
            layoutFlipHorizontally.addStretch()
            if uicfg.get('disable_flip_horizontally'):
                self.flipHorizontaliyCheckButton.setEnabled(False)
                self.flipHorizontallyLabel.setEnabled(False)
                if not self._device.isProperty('flipx'):
                    self.flipHorizontaliyCheckButton.setVisible(False)
                    self.flipHorizontallyLabel.setVisible(False)

        # flip image/data vertically
        layoutFlipVertically = None
        self.flipVerticallyCheckButton = self.flipVerticallyLabel = None
        if self._device.isProperty('flipy') or \
                not uicfg.get('disable_flip_vertically'):
            value = False
            if self._device.isProperty('flipy'):
                value = self._device.get('flipy')
            self.flipVerticallyCheckButton = common.QCheckablePushButton()
            self.flipVerticallyCheckButton.toggled.connect(
                self._updateFlipVertically)
            self.flipVerticallyLabel = QLabel(
                QCoreApplication.translate(
                    'FpaHubModule',
                    'Flip image vertically'
                )
            )
            layoutFlipVertically = QHBoxLayout()
            layoutFlipVertically.addWidget(self.flipVerticallyCheckButton)
            layoutFlipVertically.addWidget(self.flipVerticallyLabel)
            layoutFlipVertically.addStretch()
            if uicfg.get('disable_flip_vertically'):
                self.flipVerticallyCheckButton.setEnabled(False)
                self.flipVerticallyLabel.setEnabled(False)

        # auto exposure action
        self.autoExposurePushButton = QPushButton(
            QCoreApplication.translate(
                'FpaHubModule', 'Auto exposure'))
        self.autoExposurePushButton.setToolTip(
            QCoreApplication.translate(
                'FpaHubModule',
                'Adjust the exposure time automatically.')
        )
        self.autoExposurePushButton.clicked.connect(
            lambda state: self.autoExposure())

        # auto exposure action configuration dialog
        self.autoExposureConfigurePushButton = QPushButton(
            QCoreApplication.translate(
                'FpaHubModule', 'Configure Auto Exposure'))
        self.autoExposureConfigurePushButton.setIcon(
            resources.loadIcon('settings.png')
        )
        self.autoExposureConfigurePushButton.setToolTip(
            QCoreApplication.translate(
                'FpaHubModule',
                'Configure auto exposure.')
        )
        self.autoExposureConfigurePushButton.clicked.connect(
            lambda state: self.autoExposureConfigure())
        self.autoExposureConfigurationDialog = \
            autoexposure.AutoExposureConfigurationDialog(
                **uicfg['autoexposure'], parent=self
            )

        # temperature monitoring
        self.sensorTemperatureLabel = self.temperatureLivePlot = None
        if self._device.isProperty('sensortemperature'):
            self.temperatureLivePlot = plot.LivePlotx(
                None, window=None, n=100,
                initval=self._device.get('sensortemperature')
            )
            self.temperatureLivePlot.statusBar().setVisible(False)
            self.temperatureLivePlot.setAverageVisible(True)
            self.temperatureLivePlot.setLivePen('r')
            self.temperatureLivePlot.setAveragePen('g')
            self.sensorTemperatureLabel = QLabel(
                QCoreApplication.translate(
                    'FpaHubModule',
                    'Temperature sensor history'
                )
            )
            self.sensorTemperatureLabel.setBuddy(self.temperatureLivePlot)

            if 'targetsensortemperature' in self._device.properties():
                self.temperatureLivePlot.setTarget(
                    self._device.get('targetsensortemperature'))
                self.temperatureLivePlot.setTargetVisible(True)
                self.temperatureLivePlot.setTargetPen(
                    plot.mkPen(color='FFFF0080', width=2, alpha=0.5))

        # Device and preview refresh summarty.
        self.previewItemRateLabel = QLabel()
        self.deviceRateLabel = QLabel()
        rateLayout = QHBoxLayout()
        rateLayout.addWidget(self.previewItemRateLabel)
        rateLayout.addStretch()
        rateLayout.addWidget(self.deviceRateLabel)


        layout = QVBoxLayout()
        layout.addWidget(self.exposureTimeAdjustWidget)
        layout.addWidget(common.horizontalLine())
        if self.averagingAdjustWidget is not None:
            layout.addWidget(self.averagingAdjustWidget)
            layout.addWidget(common.horizontalLine())
        if self._layoutGain is not None:
            layout.addLayout(self._layoutGain)
            layout.addWidget(common.horizontalLine())
        if self.gainAdjustWidget is not None:
            layout.addWidget(self.gainAdjustWidget)
            layout.addWidget(common.horizontalLine())
        if self.offsetAdjustWidget is not None:
            layout.addWidget(self.offsetAdjustWidget)
            layout.addWidget(common.horizontalLine())
        if layoutFlipHorizontally is not None:
            layout.addLayout(layoutFlipHorizontally)
            layout.addWidget(common.horizontalLine())
        if layoutFlipVertically is not None:
            layout.addLayout(layoutFlipVertically)
            layout.addWidget(common.horizontalLine())
        layout.addWidget(self.autoExposurePushButton)
        layout.addWidget(self.autoExposureConfigurePushButton)
        layout.addWidget(common.horizontalLine())
        layout.addStretch()
        if self.temperatureLivePlot is not None:
            layout.addWidget(common.horizontalLine())
            layout.addWidget(self.sensorTemperatureLabel)
            layout.addWidget(self.temperatureLivePlot)
            layout.addWidget(common.horizontalLine())
        layout.addLayout(rateLayout)

        self._updatePreviewTimer = QTimer(parent=self)
        self._updatePreviewTimer.timeout.connect(self.fetchPreviewItem)
        refresh_period = max(int(round(uicfg['refreshperiod']*1000)), 100)
        self._updatePreviewTimer.start(refresh_period)

        self._updateTemperatureTimer = QTimer(parent=self)
        self._updatePreviewTimer.timeout.connect(self.updateTemperatures)
        self._updateTemperatureTimer.start(1000)

        self.setLayout(layout)

        self._device.acquire()

    def label(self) -> str:
        '''
        Returns a short label text that can be used as tab widget caption.
        '''
        return self._label

    def description(self) -> str:
        '''
        Returns module description.
        '''
        return self._description

    def tooltip(self) -> str:
        '''
        Returns module tooltip.
        '''
        return self._tooltip

    def icon(self):
        '''
        Returns module icon.
        '''
        return self._icon

    def _updateExposureTime(self, value):
        self._device.set('exposuretime', value)
        self.exposureTimeChanged.emit(self._device.get('exposuretime'))

    def _updateGain(self, value):
        value = self._device.property('gain').type(value)
        with self._deviceGuard:
            self._device.set('gain', value)
        if self.gainAdjustWidget is not None:
            self.gainAdjustWidget.setValue(value)
        elif self.gainCheckedWidget is not None:
            self.gainCheckedWidget.setChecked(value)

    def _updateOffset(self, value):
        value = self._device.property('offset').type(value)
        with self._deviceGuard:
            self._device.set('offset', value)
        self.offsetChanged.emit(self._device.get('offset'))

    def _updateAveraging(self, value):
        value = max(1, self._device.property('averaging').type(value))
        with self._deviceGuard:
            self._device.set('averaging', value)
        self.averagingChanged.emit(self._device.get('averaging'))

    def _updateFlipHorizontally(self, state):
        with self._deviceGuard:
            if 'flipx' in self._device.properties():
                self._device.set(
                    'flipx', self._device.property('flipx').type(state)
                )
        self.flipHorizontaliyCheckButton.setChecked(state)

    def softFlipHorizontally(self):
        return self.flipHorizontaliyCheckButton is not None and \
            not self._device.isProperty('flipx')

    def flipHorizontally(self):
        return self.flipHorizontaliyCheckButton is not None and \
            self.flipHorizontaliyCheckButton.isChecked()

    def _updateFlipVertically(self, state):
        with self._deviceGuard:
            if 'flipy' in self._device.properties():
                self._device.set(
                    'flipy', self._device.property('flipy').type(state)
                )
        self.flipVerticallyCheckButton.setChecked(state)

    def softFlipVertically(self):
        return self.flipVerticallyCheckButton is not None and \
            not self._device.isProperty('flipy')

    def flipVertically(self):
        return self.flipVerticallyCheckButton is not None and \
            self.flipVerticallyCheckButton.isChecked()

    def setPreviewRate(self, rate):
        ms_interval = int(round(1000.0*1.0/rate))
        ms_interval = clip(ms_interval, 33, 1000)
        self._updatePreviewTimer.setInterval(ms_interval)

    def previewRate(self):
        return 1.0/self._updatePreviewTimer.interval()

    def fetchPreviewItem(self):
        if self._device.isProperty('newframe'):
            item = self._device.get('newframe')
        else:
            item = self._device.get('newspectrum')

        if item is not None:
            if self.softFlipHorizontally() and self.flipHorizontally():
                item = np.fliplr(item)

            if self.softFlipVertically() and self.flipVertically():
                item = np.flipud(item)

            self._previewItemIndex += 1
            meta = self.fetchMetadata()
            self.newPreviewItem.emit((item, self._previewItemIndex, meta))

            new_timestamp = time.perf_counter()
            acquired = self._device.get('acquired')
            if self._previousTimestamp is not None:
                dt = new_timestamp - self._previousTimestamp
                previewItemRate = 1.0/dt

                self.previewItemRateLabel.setText(
                    QCoreApplication.translate(
                        'FpaHubModule', 'Preview: {:.1f} 1/s'
                    ).format(previewItemRate)
                )
                framerate = 0.0
                if self._device.isProperty('framerate'):
                    framerate = self._device.get('framerate') / \
                        self._device.get('averaging')
                else:
                    if self._previousAcquired is not None:
                        framerate = (acquired - self._previousAcquired)/dt
                        self._framerateHistory[1:] = \
                            self._framerateHistory[:-1]
                        self._framerateHistory[0] = framerate, new_timestamp
                        mask = self._framerateHistory[:, 1] >= \
                            new_timestamp - self.FRAMERATE_ESTIMATION_WINDOW
                        framerate = np.nanmean(self._framerateHistory[mask, 0])

                self.deviceRateLabel.setText(
                    QCoreApplication.translate(
                        'FpaHubModule', 'Device:{:.1f} 1/s'
                    ).format(framerate)
                )

            self._previousTimestamp = new_timestamp
            self._previousAcquired = acquired

    def fetchMetadata(self, meta=None):
        if meta is not None:
            meta = copy.deepcopy(meta)
        else:
            meta = {}

        for item in ('exposuretime', 'averaging', 'gain', 'offset',
                     'wavelength',
                     'sensortemperature', 'targetsensortemperature',
                     'width', 'height', 'x', 'y', 'bitdepth',
                     'sensorwidth', 'sensorheight', 'acquired'):
            if self._device.isProperty(item) and \
                    self._device.property(item).access.r:
                meta[item] = self._device.get(item)

        return meta

    def updateTemperatures(self, temp=None, target=None):
        if temp is None and \
                self._device.isProperty('sensortemperature'):
            with self._deviceGuard:
                temp = self._device.get('sensortemperature')
            if temp is not None:
                self.temperatureLivePlot.addSample(temp)

        if target is None and \
                self._device.isProperty('targetsensortemperature'):
            with self._deviceGuard:
                target = self._device.get('targetsensortemperature')
            if target is not None:
                self.temperatureLivePlot.setTarget(target)
            self.temperatureLivePlot.setTargetVisible(target is not None)

    def device(self):
        return self._device

    def autoExposure(self, intensity=None):
        dlg = autoexposure.AutoExposureDialog(self)
        verbose = self.autoExposureConfigurationDialog.verbose()
        if intensity is None:
            intensity = self.autoExposureConfigurationDialog.target()/100.0
        percentile = self.autoExposureConfigurationDialog.percentile()
        statistics = self.autoExposureConfigurationDialog.statistics()

        if dlg.start(self._device, intensity, maxtrials=100,
                     statistics=statistics,
                     percentile=percentile, verbose=verbose):
            self.exposureTimeAdjustWidget.setValue(dlg.result())

    def autoExposureConfigure(self):
        self.autoExposureConfigurationDialog.setWindowModality(Qt.WindowModal)
        self.autoExposureConfigurationDialog.show()

    def closeEvent(self, event):
        self._updatePreviewTimer.stop()
        self._updateTemperatureTimer.stop()
        self._device.stop()


if __name__ == '__main__':
    from widgets import common
    from basler import pylon

    app = common.prepareqt()

    camera = pylon.Pylon()
    camera.set('pixelformat', 'Mono12')

    cw = FpaHubModule(camera)

    #spectrometer = simspec.Simspec()
    #uicfg = {'ui':{'disable_flip_horizontally':True,
    #               'disable_flip_vertically':True}}
    #cw = FpaHubModule(spectrometer, cfg=uicfg)

    cw.setPreviewRate(5)
    cw.show()

    app.exec()
