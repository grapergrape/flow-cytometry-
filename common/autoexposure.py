# -*- coding: utf-8 -*-
import time

import numpy as np

from widgets import common
from common import acquisition


def clipValue(value, low, high):
    return max(min(value, high), low)

class AutoExposure:
    def __init__(self, device, tol=0.02, maxtrials=200, timeout=None,
                 statistics='max', percentile=90, mask=None,
                 acqclass=None, dialog=None, verbose=False):
        '''
        Class for automatic exposure time adjustment. The process of
        exposure time adjustment can be customized by change the values of
        the constructor parameters.

        Parameters
        ----------
        device: Device
            Data acquisition device such as camera or spectrometer that
            support exposuretime, averaging and bitdepth properties, and define
            acquire method.

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
            are also not used by the automatic exposure time adjustment method.

        acqclass: Acquisition like
            Constructs an acquisition object for the given device. Most
            device work fine with the default Acquisition class, but some
            might require a customized factuality.

        dialog: QBackgroundTask
            Dialog instance to be used with the automatic exposure time
            adjustment.

        verbose: bool
            If set to True, information on the progress of the exposure time
            adjustment is printed to stdout.
        '''
        if acqclass is None:
            acqclass = acquisition.Acquisition
        self._percentile = clipValue(percentile, 0, 100)
        self._acq_generator = acqclass
        self._dialog = dialog
        self._device = device
        self._verbose = verbose
        self._maxtrials = maxtrials
        self._statistics = statistics
        self._tol = clipValue(tol, 0.0, 1.0)
        self._intensity_range = (0, 2**device.get('bitdepth') - 1)
        self._max_intensity = self._intensity_range[1]
        self._intensity_tol = clipValue(tol, 0.0, 1.0)*self._max_intensity
        self._mask = mask
        self._max_timeout = timeout
        self._t_start = None

        if self._verbose:
            print('Identifying stuck pixels ...')
        data = self.acquireData(16, 0.0)[0]
        if data is not None:
            not_stuck_mask = data.std(0) != 0.0
            if self._verbose:
                print('Found {} stuck pixels.'.format(
                    not_stuck_mask.size - np.count_nonzero(not_stuck_mask)))

            if self._mask is not None:
                self._mask = np.logical_and(self._mask, not_stuck_mask)
            else:
                self._mask = not_stuck_mask

    def start(self, target, updateprogress=False):
        '''
        Starts the automatic exposure time adjustment using the specified
        target intensity.

        Parameters
        ----------
        target: float
            Relative target intensity from [0.0, 1.0].

        updateprogress: bool
            If True and a dialog is specified in the constructor or set by a
            call to the setDialog method, the progress of the dialog is
            updated during exposure time adjustment.

        Returns
        -------
        exposuretime: float
            If successful returns the adjusted exposure time, else
            returns None.
        '''
        tstart = time.perf_counter()

        error_details = ''
        success = False

        target = clipValue(target, 0.0, 1.0)
        target_intensity = target*self._max_intensity
        savedAcquiringSate = self._device.acquiring()
        self._device.stop()
        savedExposureTime = self._device.get('exposuretime')
        savedAveraging = self._device.get('averaging')
        self._device.set('averaging', 1)


        et_min, et_max = self._device.property('exposuretime').range

        factor = 2
        et = et_min*factor
        intensity, et = self.senseIntensity(et)
        di = np.abs(intensity - target_intensity)

        trials = 0
        while trials < self._maxtrials:
            trials += 1
            check_plus, et_plus = self.senseIntensity(et*factor, tstart=tstart)
            if check_plus is None:
                break

            di_plus = abs(target_intensity - check_plus)

            if check_plus < target_intensity/factor:
                et = et_plus
                intensity = check_plus
                di = di_plus
            else:
                check_minus, et_minus = self.senseIntensity(
                    et/factor, tstart=tstart)
                if check_minus is None:
                    break
                di_minus = abs(target_intensity - check_minus)

                if di_plus < di_minus and check_plus < self._max_intensity:
                    #if check_plus > check_minus:
                    factor = factor**0.5
                    et = et_plus
                    intensity = check_plus
                    di = di_plus
                else:
                    #if check_minus < check_plus:
                    factor = factor**0.5
                    et = et_minus
                    intensity = check_minus
                    di = di_minus

            if self._verbose:
                print('trial', trials, 'of', self._maxtrials)
                print('exposure time:', et, 'statistics: ', self._statistics,
                      'intensity/target/tolerance:',
                      intensity, '/', target_intensity,
                      '/', self._intensity_tol)
                print('factor:', factor, 'closest distance:', di)

            if di < self._intensity_tol:
                success = True
                break

            if et <= et_min and intensity > target_intensity:
                if self._dialog is not None:
                    error_details = common.QCoreApplication.translate(
                        'AutoExposure',
                        'Target intensity of {:.1f} % exceeded even when'
                        'using the shortest possible exposure time!'
                    ).format(target*100)
                else:
                    error_details = \
                        'Target intensity of {:.1f} % exceeded even when'\
                        'using the shortest possible exposure time!'.format(
                            target*100)
                break

            if et >= et_max and intensity < target_intensity:
                if self._dialog is not None:
                    error_details = common.QCoreApplication.translate(
                        'AutoExposure',
                        'Target intensity of {:.1f} % cannot be reached '
                        'even whe using the longest possible exposure time!'
                    ).format(target*100)
                else:
                    error_details = \
                        'Target intensity of {:.1f} % cannot be reached even '\
                        'when using the longest possible exposure '\
                        'time!'.format(target*100)
                break

            if factor - 1.0 < et_min/(10.0*et_max):
                # step too small ...
                break

            if self._dialog is not None and self._dialog.wasCanceled():
                break
            if self._max_timeout is not None and \
                    time.perf_counter() - tstart > self._max_timeout:
                break

            if self._dialog is not None and updateprogress:
                if self._max_timeout is not None:
                    pos = int(
                        100.0*(time.perf_counter() - tstart)/self._max_timeout
                    )
                else:
                    pos = int(100.0*trials/self._maxtrials)
                self._dialog.setProgress(min(pos, 99))

        if not success:
            # restore exposure time only if failed
            self._device.set('exposuretime', savedExposureTime)
        else:
            self._device.set('exposuretime', et)

        self._device.stop()
        self._device.set('averaging', savedAveraging)
        if savedAcquiringSate:
            self._device.acquire()

        if not success:
            if self._dialog is not None:
                if not self._dialog.wasCanceled():
                    self._dialog.reportError(
                        common.QCoreApplication.translate(
                            'AutoExposure',
                            'The automatic exposure time adjustment has '
                            'failed after {} steps!'
                        ).format(trials),
                        error_details,
                    )
            else:
                if error_details:
                    raise RuntimeError(
                        'The automatic exposure time adjustment has '
                        'failed after {} steps: {}!'.format(
                            trials, error_details)
                    )
            if self._verbose:
                print('Leaving autoexposure with result:', None)
            return None
        else:
            if self._verbose:
                print('Leaving autoexposure with result:', et)
            return et

    def setDialog(self, dialog):
        '''
        Set the dialog to be used during the automatic exposure time
        adjustment. Use None to remove any existing dialog
        '''
        self._dialog = dialog

    def acquireData(self, n, exposuretime=None, tstart=None):
        '''
        Acquires the given number of data samples (images, spectra, ...) using
        the given exposure time.

        Parameters
        ----------
        n: int
            The number of samples (images, spectra) to acquire.
        exposuretime: float
            Exposure time to use. If None, the current exposure time is used.
        tstart: float
            Timestamp since which the timeout is measured. If none, the
            current time (time.perf_counter()) is used.

        Returns
        -------
        data: ndarray
            The acquired data samples or None.
        exposuretime:
            The actual exposure time used by the device hardware.
        '''
        acquiringState = self._device.acquiring()

        if tstart is None:
            tstart = time.perf_counter()

        if exposuretime is not None:
            self._device.set('exposuretime', exposuretime)

        acq = self._acq_generator(n)
        self._device.acquire(acq)

        while 1:
            if acq.join(0.1):
                break
            if self._dialog is not None and self._dialog.wasCanceled():
                break
            if self._max_timeout is not None and \
                    time.perf_counter() - tstart > self._max_timeout:
                break

        # restore preview if required
        if acquiringState:
            self._device.acquire()

        if acq.get('acquired') >= n:
            data = acq.get('data')
            return data, self._device.get('exposuretime')

        return None, self._device.get('exposuretime')

    def senseIntensity(self, exposuretime, n=1, tstart=None):
        '''
        Acquires the specified number of samples and calculates the intensity
        using the selected statistics (see the constructor).

        Parameters
        ----------
        exposuretime: float
            Exposure time to use. If None, the current exposure time is used.
        n: int
            The number of samples (images, spectra) to acquire.
        tstart: float
            Timestamp since which the timeout is measured. If none, the
            current time (time.perf_counter()) is used.

        Returns
        -------
        intensity: ndarray
            The intensity derived from the acquired samples using the selected
            statistics.
        exposuretime:
            The actual exposure time used by the device hardware.
        '''
        if tstart is None:
            tstart = time.perf_counter()

        data, exposure_time = self.acquireData(n, exposuretime, tstart)

        if data is not None:
            if self._mask is not None:
                if data.ndim > self._mask.ndim:
                    data = data[:, self._mask]
                else:
                    data = data[self._mask]

            if self._statistics == 'max':
                data = data.max()
            elif self._statistics == 'percentile':
                data = np.percentile(data, self._percentile)
            else:
                data = data.mean()

        return data, exposure_time

    def intensityRange(self):
        '''
        Returns the intensity range of the device samples as a tuple
        (minimum, maximum).
        '''
        return (0, 2**self._device.get('bitdepth') - 1)

    def minimumIntensity(self):
        '''
        Returns the minimum intensity of the device data samples.
        '''
        return 0

    def maximumIntensity(self):
        '''
        Returns the maximum intensity of the device data samples.
        '''
        return 2**self._device.get('bitdepth') - 1

    def exposureTime(self):
        '''
        Returns the current device exposure time setting.

        Notes
        -----
        Note that this value might differ from the value used to set the
        device exposure time.
        '''
        return self._device.get('exposuretime')

    def exposureTimeRange(self):
        '''
        The valid range of exposure times supported by the device as
        a tuple (minimum, maximum).
        '''
        return self._device.property('exposuretime')

    def minimumExposureTime(self):
        '''
        The minimum exposure time supported by the device.
        '''
        return self._device.property('exposuretime').range[0]

    def maximumExposureTime(self):
        '''
        The maximum exposure time supported by the device.
        '''
        return self._device.property('exposuretime').range[1]
