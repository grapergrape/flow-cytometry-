# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget, QDialog, QComboBox, QListView, \
                            QLabel, QPushButton, \
                            QHBoxLayout, QVBoxLayout, \
                            QDoubleSpinBox, QSpinBox
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import pyqtSignal as QSignal

from widgets import common, resources
from common import autoexposure

class AutoExposureConfigurationWidget(QWidget):
    def __init__(self, statistics: str = 'max', target: float = 85.0,
                 targetrange: (float, float) = (0.0, 100.0),
                 percentile: float = 85.0,
                 percentilerange: (float, float) = (5.0, 100.0),
                 verbose: bool = False, **kwargs):
        '''
        Creates a widget that can be used to configure the parameters of
        automatic exposure adjustment.

        Parameters
        ----------
        statistics: str
            Statistics used to extract the intensity from the acquired data.
            Supported statistics include 'max' - maximum value,
            'mean' - mean value and 'percentile' - specified percentile
            (see the percentile parameter).

        target: float
            Target intensity expressed in percentage of the full range. The
            value must be from range 0.0% to 100.0%.

        targetrange: (float, float)
            The range of target intensity values in percentage of the full
            range that are allowed. Any value from 0.0% to 100.0% is
            allowe by default.

        tol: float
            Relative (normalized by maximum intensity) tolerance of the
            target intensity from (0.0, 1.0).

        maxtrials: int
            Maximum number of exposure time adjustment attempts before
            terminating the procedure.

        percentile: float
            Parameter used with the 'percentile' statistics. A value of 100
            equals 'max' statistics.

        percentilerange: (float, float)
            The allowed range of parameter percentile. Any value from 0.0 to
            100.0 is allowe by default.

        verbose: bool
            Indicates that information on the progress of the exposure time
            adjustment should be printed to stdout.

        kwargs: dict
            Parameters passed to the base class constructor.
        '''
        super().__init__(**kwargs)

        self.targetAdjust = common.QAdjustWidget()
        self.targetAdjust.setRange(int(targetrange[0]), int(targetrange[1]))
        self.targetAdjust.setValue(int(target))
        self.targetAdjust.setGotoStartVisible(True)
        self.targetAdjust.setGotoEndVisible(True)
        self.targetAdjust.setUnits('%')
        self.targetAdjust.setCaption(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'Autoexposure target intensity'
            )
        )

        self.verboseLabel = QLabel(QCoreApplication.translate(
            'AutoExposureConfigurationDialog',
            'Print details to console'))
        self.verboseCheckButton = common.QCheckablePushButton()
        self.verboseCheckButton.setChecked(bool(verbose))
        self.verboseCheckButton.setToolTip(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'Prints detailed information on the progress of the '\
                'exposure time adjustment to the console.')
        )
        self.verboseLabel.setBuddy(self.verboseCheckButton)
        verboselayout = QHBoxLayout()
        verboselayout.setContentsMargins(0, 0, 0, 0)
        verboselayout.addWidget(self.verboseCheckButton)
        verboselayout.addWidget(self.verboseLabel)
        verboselayout.addStretch()

        ae_layout_left = QVBoxLayout()
        ae_layout_left.setContentsMargins(0, 0, 0, 0)
        ae_layout_right = QVBoxLayout()
        ae_layout_right.setContentsMargins(0, 0, 0, 0)

        #layout_right.addStretch()
        ae_layout = QHBoxLayout()
        ae_layout.addLayout(ae_layout_left)
        ae_layout.addLayout(ae_layout_right)
        #auto exposure kind
        self.statisticsLabel = QLabel(
            QCoreApplication.translate('AutoExposureConfigurationDialog',
                                       'Autoexposure intensity statistics'))
        self.statisticsCombobox = QComboBox()
        self.statisticsCombobox.setToolTip(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'Statistics used by the autoexposure to calculate the '\
                'target spectrum intensity.'))
        # required for stylesheet to work #!!!!
        self.statisticsCombobox.setView(QListView())
        self.statisticsCombobox.addItem(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog', 'Average'), 'mean')
        self.statisticsCombobox.addItem(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog', 'Maximum'), 'max')
        self.statisticsCombobox.addItem(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog', 'Percentile'), 'percentile')
        self.statisticsLabel.setBuddy(self.statisticsCombobox)
        self.statisticsCombobox.currentIndexChanged.connect(
            self._statisticsChanged)

        ae_layout_left.addWidget(self.statisticsCombobox)
        ae_layout_right.addWidget(self.statisticsLabel)

        # autoexposure percentile value
        self.percentileLabel = QLabel(
            QCoreApplication.translate('AutoExposureConfigurationDialog',
                                       'Percentile'))
        self.percentileSpinbox = QDoubleSpinBox()
        self.percentileSpinbox.setToolTip(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'Value used with the percentile statistics.'))
        self.percentileSpinbox.setSingleStep(0.1)
        self.percentileSpinbox.setRange(
            float(percentilerange[0]), float(percentilerange[1]))
        self.percentileSpinbox.setValue(float(percentile))
        self.percentileLabel.setBuddy(self.percentileSpinbox)
        self.percentileSpinbox.setEnabled(False)

        ae_layout_left.addWidget(self.percentileSpinbox)
        ae_layout_right.addWidget(self.percentileLabel)

        self.setStatistics(statistics)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.targetAdjust)
        layout.addLayout(ae_layout)
        layout.addLayout(verboselayout)

        self.setLayout(layout)

    def _statisticsChanged(self, index):
        self.percentileSpinbox.setEnabled(index == 2)
        self.percentileLabel.setEnabled(index == 2)

    def percentile(self) -> float:
        '''
        Returns the current percentile value.

        Returns
        -------
        percentile: float
            Current percentile value.
        '''
        return self.percentileSpinbox.value()

    def setPercentile(self, value):
        '''
        Sets the current percentile value.

        Parameters
        ----------
        percentile: float
            New percentile value.
        '''
        self.percentileSpinbox.setValue(value)

    def target(self) -> float:
        '''
        Returns the current target intensity value.

        Returns
        -------
        target: float
            Current target intensity.
        '''
        return self.targetAdjust.value()

    def setTarget(self, value):
        '''
        Set the current target intensity value.

        Parameters
        ----------
        value: float
            New value for target intensity.
        '''
        self.targetAdjust.setValue(value)

    def verbose(self) -> bool:
        '''
        Returns the current state of the verbose mode.

        Returns
        -------
        target: bool
            Current target intensity.
        '''
        return self.verboseCheckButton.isChecked()

    def setVerbose(self, value: bool):
        '''
        Sets the current state of the verbose mode.

        Parameters
        ----------
        value: bool
            New value for the state of the verbose mode.
        '''
        self.verboseCheckButton.setChecked(value)

    def statistics(self) -> str:
        '''
        Returns the current identifier of the intensity statistics.

        Returns
        -------
        statistics: str
            Current value of the intensity statistics.
        '''
        return self.statisticsCombobox.currentData()

    def setStatistics(self, value: str):
        '''
        Sets the current intensity statistics to the specified value.

        Parameters
        ----------
        value: str
            New value of the intensity statistics.
        '''
        value = str(value).lower()
        for index in range(self.statisticsCombobox.count()):
            if value == self.statisticsCombobox.itemData(index):
                self.statisticsCombobox.setCurrentIndex(index)


class AutoExposureConfigurationDialog(QDialog):
    def __init__(self, title=None, parent=None, **kwargs):
        '''
        Creates a modal dialog for configuring the parameters of
        automatic exposure adjustment. Use to exec_() method to show
        the modal dialog.

        All the methods defined by the AutoExposureConfigurationWidget
        class are directly accessable from the dialog object.

        Parameters
        ----------
        title: str
            Optional dialog title.

        kwargs: dict
            Arguments passed to AutoExposureConfigurationWidget.
        '''

        super().__init__(parent=parent)

        if title is None:
            title = QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'Autoexposure Dialog'
            )
        self.setWindowTitle(title)

        self.autoExposureConfigurationWidget = AutoExposureConfigurationWidget(
            **kwargs
        )

        self.okButton = QPushButton(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'OK'
            )
        )
        self.okButton.setIcon(resources.loadIcon('ok.png'))
        self.okButton.clicked.connect(self.accept)

        self.cancelButton = QPushButton(
            QCoreApplication.translate(
                'AutoExposureConfigurationDialog',
                'Cancel'
            )
        )
        self.cancelButton.setIcon(resources.loadIcon('cancel.png'))
        self.cancelButton.clicked.connect(self.reject)

        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.okButton)
        buttonLayout.addWidget(self.cancelButton)

        layout = QVBoxLayout()
        layout.addWidget(self.autoExposureConfigurationWidget)
        layout.addStretch()
        layout.addLayout(buttonLayout)
        self.setLayout(layout)

    def __getattr__(self, attr):
        item = getattr(self.autoExposureConfigurationWidget, attr)
        return item


class AutoExposureDialog(common.QBackgroundTask):
    def __init__(self, parent=None):
        '''
        Automatically adjusts the exposure time so that the maximum intensity
        of the brightest pixels is at the specified level - within the
        specified tolerances. Once complete, the optimal exposure time can be
        retrieved by calling the result method.

        The adjustmen of exposure time is conducted in a background thread. A
        progress dialog is shown that allows user to interrupt the
        process.

        Once the object is constructed, call start method to start the
        exposure time adjustment.

        An instance of AutoExposure class from module autoexposure is
        used to handle the exposure time adjustment.
        '''
        common.QBackgroundTask.__init__(self, parent=parent)

    def start(self, device, intensity=0.85,
              tol=0.02, maxtrials=200, timeout=None,
              statistics='max', percentile=90, mask=None,
              acqclass=None, verbose=False):
        '''
        Starts automatic exposure time adjustment. Once complete, the
        optimal exposure time can be retrieved by calling the
        result method.

        Parameters
        ----------
        device: Device
            Data acquisition device such as camera or spectrometer that
            support exposuretime and bitdepth properties, and define
            acquire method.

        intensity: float
            Target (relative) intensity of the data samples at the
            optimal exposure time computed by the selected statistics.

        tol: float
            Relative (normalized by maximum intensity) tolerance of the
            target intensity from (0.0, 1.0).

        maxtrials: int
            Maximum number of exposure time adjustment attempts before
            terminating the procedure.

        timeout: float
            Maximum time (s) available to the exposure time adjustment method.
            The procedure is aborted if exposure time that satisfies the
            requirements is not found within the available time.

        statistics: str
            Statistics used to extract the intensity from the acquired data.
            Supported statistics include 'max' - maximum value,
            'mean' - mean value and 'percentile' - specified percentile
            (see the percentile parameter).

        percentile: float
            Parameter used with the 'percentile' statistics. A value of 100
            equals 'max' statistics.

        mask: ndarray bool
            A boolean mask of pixels that are used by the automatic exposure
            time adjustment. Regardless of the value of this parameter,
            all pixels that are stuck (std = 0) at the minimum exposure time
            are also not used by automatic exposure time adjustment method.abs

        acqclass: Acquisition like
            Constructs an acquisition object for the given device. Most
            device work fine with the default Acquisition class, but some
            might require a customized subclass.

        verbose: bool
            If set to True, information on the progress of the exposure time
            adjustment is printed to stdout.abs

        Returns
        -------
            True on success, False otherwise.
        '''
        ae = autoexposure.AutoExposure(
            device, tol, maxtrials, timeout, statistics, percentile, mask,
            acqclass, self, verbose)

        return common.QBackgroundTask.start(
            self,
            target=ae.start,
            args=(intensity, True),
            title=QCoreApplication.translate(
                'AutoExposureDialog',
                'Automatic exposure time adjustment'
            ),
            label=QCoreApplication.translate(
                'AutoExposureDialog',
                'Adjusting the exposure time.'
            )
        )


class AutoExposureTimeWidget(QWidget):
    exposureTimeChanged = QSignal(float)

    AE_MAX_STATISTICS = 'max'
    AE_PERCENTILE_STATISTICS = 'percentile'
    AE_MEAN_STATISTICS = 'mean'

    AE_ALL_STATISTICS = [AE_MAX_STATISTICS, AE_PERCENTILE_STATISTICS,
                         AE_MEAN_STATISTICS]
    @staticmethod
    def index2statistics(index):
        return {0:AutoExposureTimeWidget.AE_MAX_STATISTICS,
                1:AutoExposureTimeWidget.AE_PERCENTILE_STATISTICS,
                2:AutoExposureTimeWidget.AE_MEAN_STATISTICS}[index]

    @staticmethod
    def statistics2index(statistics):
        return {AutoExposureTimeWidget.AE_MAX_STATISTICS:0,
                AutoExposureTimeWidget.AE_PERCENTILE_STATISTICS:1,
                AutoExposureTimeWidget.AE_MEAN_STATISTICS:2}[statistics]

    def __init__(self, device, parent=None, title=None):
        QWidget.__init__(self, parent)

        if title is not None:
            self.setWindowTitle(title)

        self._device = device

        etp = device.property('exposuretime')
        self.exposureTimeAdjust = common.QAdjustWidget()
        self.exposureTimeAdjust.setGotoStartVisible(True)
        self.exposureTimeAdjust.setGotoEndVisible(True)
        self.exposureTimeAdjust.setSliderToolTip = QLabel(
            QCoreApplication.translate('AutoExposureTimeWidget',
                                       'Adjust the exposure time.'))
        self.exposureTimeAdjust.setTransform(
            common.QAdjustWidget.FpLogTransform(
                etp.range[0], etp.range[1], 1000))
        self.exposureTimeAdjust.setUnits('s')
        self.exposureTimeAdjust.setCaption(
            QCoreApplication.translate(
                'AutoExposureTimeWidget', 'Exposure time'))
        self.exposureTimeAdjust.valueChanged.connect(
            self.exposureTimeChanged.emit)

        self.autoExposurePushButton = QPushButton(
            QCoreApplication.translate(
                'AutoExposureTimeWidget', 'Auto exposure'))
        self.autoExposurePushButton.setToolTip(
            QCoreApplication.translate(
                'AutoExposureTimeWidget',
                'Adjust the exposure time automatically using 95% target.')
        )
        self.autoExposurePushButton.clicked.connect(
            lambda state: self.autoExposure())

        self.autoexposureTargetAdjust = common.QAdjustWidget()
        self.autoexposureTargetAdjust.setRange(0, 100)
        self.autoexposureTargetAdjust.setValue(85)
        self.autoexposureTargetAdjust.setGotoStartVisible(True)
        self.autoexposureTargetAdjust.setGotoEndVisible(True)
        self.autoexposureTargetAdjust.setUnits('%')
        self.autoexposureTargetAdjust.setCaption(
            QCoreApplication.translate(
                'AutoExposureTimeWidget',
                'Autoexposure target intensity'
            )
        )

        self.autoexposureVerboseLabel = QLabel(QCoreApplication.translate(
            'AutoExposureTimeWidget',
            'Print details to console'))
        self.autoexposureVerboseCheckButton = common.QCheckablePushButton()
        self.autoexposureVerboseCheckButton.setToolTip(
            QCoreApplication.translate(
                'AutoExposureTimeWidget',
                'Prints detailed information on the progress of the '\
                'exposure time adjustment to the console.')
        )
        self.autoexposureVerboseLabel.setBuddy(
            self.autoexposureVerboseCheckButton)
        verboselayout = QHBoxLayout()
        verboselayout.setContentsMargins(0, 0, 0, 0)
        verboselayout.addWidget(self.autoexposureVerboseCheckButton)
        verboselayout.addWidget(self.autoexposureVerboseLabel)
        verboselayout.addStretch()

        ae_layout_left = QVBoxLayout()
        ae_layout_left.setContentsMargins(0, 0, 0, 0)
        ae_layout_right = QVBoxLayout()
        ae_layout_right.setContentsMargins(0, 0, 0, 0)

        #layout_right.addStretch()
        ae_layout = QHBoxLayout()
        ae_layout.addLayout(ae_layout_left)
        ae_layout.addLayout(ae_layout_right)
        #auto exposure kind
        self.autoexposureKindLabel = QLabel(
            QCoreApplication.translate('AutoExposureTimeWidget',
                                       'Autoexposure intensity statistics'))
        self.autoexposureKindCombobox = QComboBox()
        self.autoexposureKindCombobox.setToolTip(
            QCoreApplication.translate(
                'AutoExposureTimeWidget',
                'Statistics used by the autoexposure to calculate the '\
                'target image intensity.'))
        self.autoexposureKindCombobox.setView(QListView()) # required for stylesheet to work #!!!!
        self.autoexposureKindCombobox.addItem(
            QCoreApplication.translate(
                'AutoExposureTimeWidget', 'Maximum'))
        self.autoexposureKindCombobox.addItem(
            QCoreApplication.translate(
                'AutoExposureTimeWidget', 'Percentile'))
        self.autoexposureKindCombobox.addItem(
            QCoreApplication.translate(
                'AutoExposureTimeWidget', 'Mean'))
        self.autoexposureKindCombobox.setCurrentIndex(0)
        self.autoexposureKindLabel.setBuddy(self.autoexposureKindCombobox)
        self.autoexposureKindCombobox.currentIndexChanged.connect(
            self._autoexposureStatisticsChanged)

        ae_layout_left.addWidget(self.autoexposureKindCombobox)
        ae_layout_right.addWidget(self.autoexposureKindLabel)

        # autoexposure percentile value
        self.autoexposurePercentileLabel = QLabel(
            QCoreApplication.translate('AutoExposureTimeWidget',
                                       'Percentile'))
        self.autoexposurePercentileSpinbox = QDoubleSpinBox()
        self.autoexposurePercentileSpinbox.setToolTip(
            QCoreApplication.translate(
                'AutoExposureTimeWidget',
                'Value used with the percentile statistics.'))
        self.autoexposurePercentileSpinbox.setSingleStep(0.1)
        self.autoexposurePercentileSpinbox.setRange(0, 100)
        self.autoexposurePercentileSpinbox.setValue(90)
        self.autoexposurePercentileLabel.setBuddy(
            self.autoexposurePercentileSpinbox)
        self.autoexposurePercentileSpinbox.setEnabled(False)

        ae_layout_left.addWidget(self.autoexposurePercentileSpinbox)
        ae_layout_right.addWidget(self.autoexposurePercentileLabel)


        self.autoexposureVerboseLabel = QLabel(QCoreApplication.translate(
            'CalibrationConfigurationWidget',
            'Print details to console'))
        self.autoexposureVerboseCheckButton = common.QCheckablePushButton()
        self.autoexposureVerboseCheckButton.setToolTip(
            QCoreApplication.translate(
                'CalibrationConfigurationWidget',
                'Prints detailed information on the progress of the '\
                'exposure time adjustment to the console.')
        )
        self.autoexposureVerboseLabel.setBuddy(
            self.autoexposureVerboseCheckButton)
        verboselayout = QHBoxLayout()
        verboselayout.setContentsMargins(0, 0, 0, 0)
        verboselayout.addWidget(self.autoexposureVerboseCheckButton)
        verboselayout.addWidget(self.autoexposureVerboseLabel)
        verboselayout.addStretch()

        # averaging percentile value
        avgp = device.property('averaging')
        self.averagingLabel = QLabel(
            QCoreApplication.translate('AutoExposureTimeWidget',
                                       'Averaging'))
        self.averagingSpinbox = QSpinBox()
        self.averagingSpinbox.setToolTip(
            QCoreApplication.translate(
                'AutoExposureTimeWidget',
                'Number of data samples in the average.'))
        self.averagingSpinbox.setRange(1, avgp.range[1])
        self.averagingSpinbox.setValue(1)
        self.averagingLabel.setBuddy(self.averagingSpinbox)
        self.averagingSpinbox.valueChanged.connect(self._setDeviceAveraging)
        averaging_layout_left = QVBoxLayout()
        averaging_layout_left.setContentsMargins(0, 0, 0, 0)
        averaging_layout_right = QVBoxLayout()
        averaging_layout_right.setContentsMargins(0, 0, 0, 0)
        averaging_layout_left.addWidget(self.averagingSpinbox)
        averaging_layout_right.addWidget(self.averagingLabel)

        averaging_layout = QHBoxLayout()
        averaging_layout.addLayout(averaging_layout_left)
        averaging_layout.addLayout(averaging_layout_right)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.exposureTimeAdjust)
        layout.addLayout(averaging_layout)
        layout.addWidget(common.horizontalLine())
        layout.addLayout(ae_layout)
        layout.addWidget(self.autoExposurePushButton)
        layout.addWidget(self.autoexposureTargetAdjust)
        layout.addLayout(verboselayout)

        self.setLayout(layout)

        self.exposureTimeAdjust.valueChanged.connect(
            self._setDeviceExposureTime)

        self.exposureTimeAdjust.valueChanged.emit(self.exposureTime())
        self.averagingSpinbox.valueChanged.emit(self.averaging())

        self._opt_size = self.adjustSize()

    def _setDeviceExposureTime(self, value):
        # print('Setting device exposure time to:', value)
        self._device.set('exposuretime', value)

    def _setDeviceAveraging(self, value):
        # print('Setting device exposure time to:', value)
        acquiring = self._device.acquiring()
        self._device.stop()
        self._device.set('averaging', value)
        if acquiring:
            self._device.acquire()

    def _autoexposureStatisticsChanged(self, index):
        self.autoexposurePercentileSpinbox.setEnabled(index == 1)
        self.autoexposurePercentileLabel.setEnabled(index == 1)

    def autoExposure(self, intensity=None):
        dlg = AutoExposureDialog(self)
        verbose = self.autoexposureVerboseCheckButton.isChecked()
        if intensity is None:
            intensity = self.autoexposureTargetAdjust.value()/100.0
        # debug
        # from pybox.common import autoexposure
        # ae = autoexposure.AutoExposure(
        #     self._device, statistics=self.autoexposureStatistics(),
        #     percentile=self.percentile(), verbose=verbose)
        # ae.start(intensity)
        if verbose:
            et_start = self._device.get('exposuretime')
        dlg_result = dlg.start(self._device, intensity, maxtrials=100,
                               statistics=self.autoexposureStatistics(),
                               percentile=self.percentile(), verbose=verbose)
        if verbose:
            print('Autoexposure dialog result:', dlg_result)
            print('Device exposure time before autoexposure:', et_start)
        if dlg_result:
            et = dlg.result()
            self.exposureTimeAdjust.setValue(et)
            if verbose:
                print('Auto exposure result:', et)
            if verbose:
                et = self._device.get('exposuretime')
                print('Device exposure time:', et)

    def averaging(self):
        return self.averagingSpinbox.value()

    def setAveraging(self, averaging):
        self.averagingSpinbox.setValue(averaging)

    def exposureTime(self):
        return self.exposureTimeAdjust.value()

    def setExposureTime(self, exposuretime):
        self.exposureTimeAdjust.setValue(exposuretime)

    def autoexposureStatistics(self):
        return AutoExposureTimeWidget.index2statistics(
            self.autoexposureKindCombobox.currentIndex())

    def setAutoexposureStatistics(self, statistics):
        index = AutoExposureTimeWidget.statistics2index(statistics)
        self.autoexposureKindCombobox.setCurrentIndex(index)

    def percentile(self):
        return self.autoexposurePercentileSpinbox.value()

    def setPercentile(self, percentile):
        self.autoexposurePercentileSpinbox.setValue(percentile)

    def autoexposureIntensity(self):
        return self.autoexposureTargetAdjust.value()

    def setAutoexposureIntensity(self, intensity):
        self.autoexposureTargetAdjust.setValue(intensity)

    def autoexposure(self, intensity=None):
        if intensity is None:
            intensity = self.autoexposureIntensity()

        dlg = AutoExposureDialog(self)
        result = dlg.start(self._device, intensity,
                           statistics=self.autoexposureStatistics(),
                           percentile=self.percentile())
        if result:
            self.setExposureTime(dlg.result())

if __name__ == '__main__':
    import sys
    from widgets import common

    app = common.prepareqt()

    aetCfg = AutoExposureConfigurationDialog('Auto exposure configuration')
    aetCfg.setTarget(80.0)
    aetCfg.show()

    sys.exit(app.exec())
