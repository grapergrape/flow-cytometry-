import threading
import ctypes
import sys
import math
import os
import platform

import numpy as np

import common.device as device
import common.acquisition as acquisition

BIN_PATH = 'bin'

class Api:
    PATH = os.path.dirname(__file__)

    def __init__(self):
        if platform.system() == 'Linux':
            if math.log2(sys.maxsize) > 32:
                dll = ctypes.cdll.LoadLibrary(
                    os.path.join(BIN_PATH, 'pybasler64.so'))
            else:
                dll = ctypes.cdll.LoadLibrary(
                    os.path.join(BIN_PATH, 'pybasler32.so'))

        elif platform.system() == 'Windows':
            if math.log2(sys.maxsize) > 32:
                dll = ctypes.cdll.LoadLibrary(
                    os.path.join(BIN_PATH, 'pybasler64.dll'))
            else:
                dll = ctypes.cdll.LoadLibrary(
                    os.path.join(BIN_PATH, 'pybasler32.dll'))
        else:
            raise RuntimeError('Your operating system is not supported'\
                               'at the moment.\n'\
                               'Program is tested only on Windows and Linux'\
                               'operating systems!')

        self.callback = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)

        self.initializePylon = dll.initializePylon
        self.initializePylon.argtypes = []
        self.initializePylon.restype = None

        self.terminatePylon = dll.terminatePylon
        self.terminatePylon.argtypes = []
        self.terminatePylon.restype = None

        self.findDevices = dll.findDevices
        self.findDevices.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.findDevices.restype = ctypes.c_size_t

        self.create = dll.create
        self.create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.create.restype = ctypes.c_void_p

        self.destroy = dll.destroy
        self.destroy.argtypes = [ctypes.c_void_p]

        self.isConnected = dll.isConnected
        self.isConnected.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.isConnected.restype = ctypes.c_int

        self.flags = dll.flags
        self.flags.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                               ctypes.POINTER(ctypes.c_int)]
        self.flags.restype = ctypes.c_int

        self.isImplemented = dll.isImplemented
        self.isImplemented.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                       ctypes.POINTER(ctypes.c_int)]
        self.isImplemented.restype = ctypes.c_int

        self.isAvailable = dll.isAvailable
        self.isAvailable.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.POINTER(ctypes.c_int)]
        self.isAvailable.restype = ctypes.c_int

        self.isWritable = dll.isWritable
        self.isWritable.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_int)]
        self.isWritable.restype = ctypes.c_int

        self.isReadable = dll.isReadable
        self.isReadable.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_int)]
        self.isReadable.restype = ctypes.c_int

        self.commandExecute = dll.commandExecute
        self.commandExecute.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.commandExecute.restype = ctypes.c_int

        self.commandDone = dll.commandDone
        self.commandDone.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.POINTER(ctypes.c_int)]
        self.commandDone.restype = ctypes.c_int

        self.getlasterror = dll.getlasterror
        self.getlasterror.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.getlasterror.restype = ctypes.c_int

        self.acquire = dll.acquire
        self.acquire.argtypes = \
            [ctypes.c_void_p, ctypes.c_size_t, self.callback]
        self.acquire.restype = ctypes.c_int

        self.isacquiring = dll.isacquiring
        self.isacquiring.argtypes = \
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.isacquiring.restype = ctypes.c_int

        self.stop = dll.stop
        self.stop.argtypes = [ctypes.c_void_p]
        self.stop.restype = ctypes.c_int

        self.hasbool = dll.hasbool
        self.hasbool.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.hasbool.restype = ctypes.c_int

        self.getbool = dll.getbool
        self.getbool.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.getbool.restype = ctypes.c_int

        self.setbool = dll.setbool
        self.setbool.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.setbool.restype = ctypes.c_int

        self.hasint = dll.hasint
        self.hasint.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.hasint.restype = ctypes.c_int

        self.getint = dll.getint
        self.getint.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64)]
        self.getint.restype = ctypes.c_int

        self.setint = dll.setint
        self.setint.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64]
        self.setint.restype = ctypes.c_int

        self.intrange = dll.intrange
        self.intrange.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p,
             ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]
        self.intrange.restype = ctypes.c_int

        self.hasfloat = dll.hasfloat
        self.hasfloat.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.hasfloat.restype = ctypes.c_int

        self.getfloat = dll.getfloat
        self.getfloat.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        self.getfloat.restype = ctypes.c_int

        self.setfloat = dll.setfloat
        self.setfloat.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        self.setfloat.restype = ctypes.c_int

        self.floatrange = dll.floatrange
        self.floatrange.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p,
             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.floatrange.restype = ctypes.c_int

        self.hasenum = dll.hasenum
        self.hasenum.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.hasenum.restype = ctypes.c_int

        self.getenum = dll.getenum
        self.getenum.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.getenum.restype = ctypes.c_int

        self.setenum = dll.setenum
        self.setenum.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.setenum.restype = ctypes.c_int

        self.hasstr = dll.hasstr
        self.hasstr.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.hasstr.restype = ctypes.c_int

        self.getstr = dll.getstr
        self.getstr.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.getstr.restype = ctypes.c_int

        self.setstr = dll.setstr
        self.setstr.argtypes = \
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.setstr.restype = ctypes.c_int

        self.initializePylon()

    def __del__(self):
        self.terminatePylon()

_pixelFormats = {
    'Mono8':8,
    'Mono10':10,
    'Mono10p':10,
    'Mono12':12,
    'Mono12p':12,
    'BayerGR8':8,
    'BayerRG8':8,
    'BayerGB8':8,
    'BayerBG8':8,
    'BayerGR10':10,
    'BayerGR10p':10,
    'BayerRG10':10,
    'BayerRG10p':10,
    'BayerGB10':10,
    'BayerGB10p':10,
    'BayerBG10':10,
    'BayerBG10p':10,
    'BayerGR12':12,
    'BayerGR12p':12,
    'BayerRG12':12,
    'BayerRG12p':12,
    'BayerGB12':12,
    'BayerGB12p':12,
    'BayerBG12':12,
    'BayerBG12p':12,
    'RGB8':8,
    'BGR8':8,
    'YCbCr422_8':8,
}

class Configuration(object):
    def __init__(self, serial='', numbuffers=10):
        if isinstance(serial, Configuration):
            cfg = serial
            self._serial = cfg.serial
            self._numbuffers = cfg.numbuffers
        else:
            self._serial = str(serial)
            self._numbuffers = max(int(numbuffers), 1)

    def _get_serial(self):
        return self._serial
    serial = property(_get_serial, None, None, 'Device serial number.')

    def _get_numbuffers(self):
        return self._numbuffers
    numbuffers = property(_get_numbuffers, None, None,
                          'Numner of frames in the frame buffer.')

    def __repr__(self):
        return "Configuration(serial='{}', numbuffers={})".format(
            self._serial, self._numbuffers)

    def __str__(self):
        return self.__str__() + \
            ' # Configuration object at 0x{:>08X}.'.format(id(self))

class Pylon(device.Device):

    @staticmethod
    def findEx():
        result = []
        api = api = Api()

        buffer = ctypes.create_string_buffer(8192)
        n = ctypes.c_size_t(8192)
        n_found = api.findDevices(buffer, n)

        descriptors = buffer.value.decode('utf8').strip('||')

        if n_found > 0:
            devs = descriptors.split('||')
            for dev in devs:
                properties = {}
                # "<name>::<value>;; ... ||<name>::<value>;; ... || ..."
                entries = dev.split(';;')
                for entry in entries:
                    pair = entry.split('::')
                    if pair and len(pair) == 2:
                        name, value = pair
                        properties[name] = value
                if properties:
                    result.append(properties)
        return result

    @staticmethod
    def find():
        return Pylon.findEx()

    def __init__(self, cfg=Configuration()):

        #base class constructor
        device.Device.__init__(self, device.Descriptor('Pylon', \
            'Basler Pylon compatible imaging device driver', \
            'Basler', cfg.serial, -1))

        self._frame = self._averageFrame = None
        self._averaging = 1
        self._ninaverage = 0

        self._api = api = Api()

        self._device = api.create(
            ctypes.c_char_p(cfg.serial.encode('ascii')),
            ctypes.c_int(cfg.numbuffers))
        if self._device is None:
            raise ValueError(
                'Pylon device not found or could not be connected!')

        self._cfg = Configuration(cfg)

        # access guard
        self._getsetRLock = threading.RLock()
        self._previewRLock = threading.RLock()

        self._cAcquisitionCallback = api.callback(self._acquisitionCallback)
        self._acquisitionerror = None
        self._acquisition = None

        widthrange = self._intrange('Width')
        heightrange = self._intrange('Height')
        xrange = self._intrange('X')
        yrange = self._intrange('Y')
        horizontalBinningRange = self._intrange('BinningHorizontal')
        verticalBinningRange = self._intrange('BinningVertical')
        exposuretimerange = self._floatrange('ExposureTime')
        exposuretimerange = (exposuretimerange[0]*1e-6, exposuretimerange[1]*1e-6,)
        gainrange = self._floatrange('Gain')
        targetframeraterange = self._floatrange('AcquisitionFrameRate')
        digitaloffsetrange = self._floatrange('BlackLevel')

        self._propertynamelut = {
            'pixelformat':'PixelFormat', 'bitdepth':'PixelSize',
            'sensorwidth':'SensorWidth', 'sensorheight':'SensorHeight',
            'width':'Width', 'height':'Height', 'x':'OffsetX', 'y':'OffsetY',

            'horizontalbinning':'BinningHorizontal',
            'verticalbinning':'BinningVertical',
            'horizontalbinningmode':'BinningHorizontalMode',
            'verticalbinningmode':'BinningVerticalMode',
            'gainauto': 'GainAuto',
            'exposuretime':'ExposureTime',
            'exposuremode':'ExposureMode',
            'exposureauto':'ExposureAuto',
            'offsetselector':'BlackLevelSelector',
            'offset':'BlackLevel',
            'gainselector':'GainSelector', 'gain':'Gain',
            'enabletargetframerate':'AcquisitionFrameRateEnable',
            'targetframerate':'AcquisitionFrameRate',
            'framerate':'ResultingFrameRate',

            'acquisitionmode':'AcquisitionMode',
            'burstframecount':'AcquisitionBurstFrameCount',
            'triggerselector':'TriggerSelector',
            'triggersource':'TriggerSource',
            'triggeractivation':'TriggerActivation',
            'triggermode':'TriggerMode',

            'triggersoftware':'TriggerSoftware',

            'softwaresignalpulse':'SoftwareSignalPulse',
            'softwaresignalselector':'SoftwareSignalSelector',

            'iolineselector':'LineSelector',
            'iolinemode':'LineMode',
            'iolinesource':'LineSource',

            'name':'DeviceModelName',
            'vendor':'DeviceVendorName',
            'serial':'DeviceSerialNumber',
            'version':'DeviceVersion',
            'firmware':'DeviceFirmwareVersion',

            'newframe':'new',
            'lastframe':'last',
            }

        self.addProperties(
            (
                device.Property(
                    str, 'name', 'Device name.',
                    'r', 1),
                device.Property(
                    str, 'vendor', 'Device vendor name.',
                    'r', 1),
                device.Property(
                    str, 'serial', 'Device serial number.',
                    'r', 1),
                device.Property(
                    str, 'version', 'Device version.',
                    'r', 1),
                device.Property(
                    str, 'firmware', 'Device firmware version.',
                    'r', 1),

                device.Property(
                    np.ndarray, 'lastframe',
                    'Last acquired preview frame if available.',
                    'r', 1, None, None),
                device.Property(
                    np.ndarray, 'newframe',
                    'A new preview frame if available.',
                    'r', 1, None, None),
                device.Property(
                    int, 'acquired',
                    'Number of acquired frames since the acquisition start.',
                    'r', 1, 0, (0, sys.maxsize)),

                device.Property(
                    int, 'averaging', 'Frame averaging.',
                    'rw', 1, 1, (1, 65535)),
                device.Property(
                    int, 'bitdepth', 'Pixel bit depth - only utilized bits.',
                    'r', 1),
                device.Property(
                    str, 'pixelformat', 'Pixel format.',
                    'rw', 1, 'Mono12', ('Mono8', 'Mono12')),
                device.Property(
                    int, 'sensorwidth', 'Sensor width (pixels).',
                    'r', 1),
                device.Property(
                    int, 'sensorheight', 'Sensor height (pixels).',
                    'r', 1),
                device.Property(
                    int, 'width', 'Region of interest width (pixels).',
                    'rw', 1, widthrange[1], widthrange),
                device.Property(
                    int, 'height', 'Region of interest height (pixels).',
                    'rw', 1, heightrange[1], heightrange),
                device.Property(
                    int, 'x',
                    'Horizontal offset of the region of interest '\
                    'from the top left corner (pixels).',
                    'rw', 1, xrange[1], xrange),
                device.Property(
                    int, 'y',
                    'Vertical offset of the region of interest '\
                    'from the top left corner (pixels).',
                    'rw', 1, yrange[1], yrange),

                device.Property(
                    int, 'horizontalbinning',
                    'Horizontal (x axis) binning of the pixels.',
                    'rw', 1, 1, horizontalBinningRange),
                device.Property(
                    int, 'verticalbinning',
                    'Vertical (y axis) binning of the pixels.',
                    'rw', 1, 1, verticalBinningRange),

                device.Property(
                    str, 'horizontalbinningmode',
                    'Horizontal (x axis) pixel binning mode.',
                    'rw', 1, 'sum', ['sum', 'average']),
                device.Property(
                    str, 'verticalbinningmode',
                    'Vertical (y axis) pixel binning mode.',
                    'rw', 1, 'sum', ['Sum', 'Average']),

                device.Property(
                    int, 'offsetselector',
                    'Dark level offset channel selector.',
                    'rw', 1, 'All', ('All',)),
                device.Property(
                    float, 'offset', 'Dark level offset (counts).',
                    'rw', 1, digitaloffsetrange[0], digitaloffsetrange),

                device.Property(
                    str, 'gainselector', 'Gain type selector.',
                    'rw', 1, 'All', ('All', 'DigitalAll', 'AnalogAll')),
                device.Property(
                    float, 'gain', 'Analog sensor gain (dB).',
                    'rw', 1, gainrange[0], gainrange),
                device.Property(
                    str, 'gainauto', 'Gain mode.',
                    'rw', 1, 'Off', ('Off', 'Once', 'Continuous')),

                device.Property(
                    float, 'exposuretime', 'Exposure time (s).',
                    'rw', 1, exposuretimerange[0], exposuretimerange),
                device.Property(
                    str, 'exposuremode', 'Exposure mode.',
                    'rw', 1, 'Timed', ('Timed', 'TriggerWidth')),
                device.Property(
                    str, 'exposureauto', 'Auto-exposure mode.',
                    'rw', 1, 'Off', ('Off', 'Once', 'Continuous')),

                device.Property(
                    str, 'acquisitionmode', 'Acquisition mode.',
                    'rw', 1, 'Continuous', ('SingleFrame', 'Continuous')),
                device.Property(
                    int, 'burstframecount',
                    'Number of frames to acquire in a burst when configured '
                    'for burst trigger mode.',
                    'rw', 1, 1),

                device.Property(
                    str, 'triggerselector', 'Select the trigger to configure.',
                    'rw', 1, 'FrameStart', ('FrameStart', 'FrameBurstStart')),
                device.Property(
                    str, 'triggermode', 'Turns on off the selected trigger.',
                    'rw', 1, 'Off', ('On', 'Off')),

                device.Property(
                    str, 'triggersource', 'Control the frame rate.',
                    'rw', 1, 'Software',
                    ('Software', 'Line1', 'Line2', 'Line3', 'Line4',
                     'SoftwareSignal1', 'SoftwareSignal2', 'SoftwareSignal3')),
                device.Property(
                    str, 'triggeractivation', 'Control the frame rate.',
                    'rw', 1, 'Rising',
                    ('RisingEdge', 'FallingEdge', 'AnyEdge',
                     'LevelHigh', 'LevelLow')),

                device.Property(
                    bool, 'triggersoftware',
                    'Generates one software trigger. '
                    'Returns trigger-done state when the property is read.',
                    'rw', 1, True),

                device.Property(
                    str, 'softwaresignalselector',
                    'Selects the active software signal line.'
                    'Returns trigger-done state when the property is read.',
                    'rw', 1, 'SoftwareSignal1',
                    ('SoftwareSignal1', 'SoftwareSignal2',
                     'SoftwareSignal3', 'SoftwareSignal4')),

                device.Property(
                    bool, 'softwaresignalpulse',
                    'Generates one software signal pulse on the selected '
                    'software signal line. '
                    'Returns signal pulse-done state when the property is read.',
                    'rw', 1, True),


                device.Property(
                    str, 'iolineselector', 'Select I/O data line.',
                    'rw', 1, 'Line1',
                    ('Line1', 'Line2', 'Line3', 'Line4')),
                device.Property(
                    str, 'iolinemode', 'Select I/O data line mode/direction.',
                    'rw', 1, 'Input',
                    ('Input', 'Output')),
                device.Property(
                    str, 'iolinesource', 'Set source signal for the I/O line.',
                    'rw', 1, 'Off',
                    ('Off', 'ExposureActive', 'FrameTriggerWait',
                     'FrameBurstTriggerWait', 'Timer1Active',
                     'UserOutput0', 'UserOutput1',
                     'UserOutput2', 'UserOutput3', 'FlashWindow')),

                device.Property(
                    bool, 'enabletargetframerate',
                    'Sets the frame rate to targetframerate if enabled.',
                    'rw', 1, False),
                device.Property(
                    float, 'targetframerate', 'Frame rate (1/s).',
                    'rw', 1, targetframeraterange[1], targetframeraterange),
                device.Property(
                    float, 'framerate', 'Frame rate (1/s).', 'r', 1),
            )
        )
        self._allocframe()
        self._preview = acquisition.Acquisition('inf')

        # update device descriptor
        self.descriptor().serial = self.get('serial')
        self.descriptor().prettyName = self.get('name')
        self.descriptor().manufacturer = self.get('vendor')

    def connected(self):
        result = ctypes.c_int(0)
        self._handleapi(
            self._api.isConnected(self._device, ctypes.byref(result)))
        return result.value != 0

    def _bitdepth2int(self, strvalue):
        return int(strvalue[3:])

    def _allocframe(self):
        shape = (self.get('height'),
                 self.get('width'),)
        nbytes = int(math.ceil(self.get('bitdepth')/8))
        nptype = {1:np.uint8, 2:np.uint16, 3:np.uint32, 4:np.uint32}[nbytes]
        if self._frame is not None:
            if (shape == self._frame.shape) and (nptype == self._frame.dtype):
                return
        self._averageFrame = np.zeros(shape, np.float64)
        self._frame = np.zeros(shape, nptype)

    def _intrange(self, prop):
        minimum = ctypes.c_int64(0)
        maximum = ctypes.c_int64(0)
        self._api.intrange(self._device, ctypes.c_char_p(prop.encode('ascii')),
                           ctypes.byref(minimum), ctypes.byref(maximum))
        return minimum.value, maximum.value

    def _floatrange(self, prop):
        minimum = ctypes.c_double(0.0)
        maximum = ctypes.c_double(0.0)
        self._api.floatrange(self._device,
                             ctypes.c_char_p(prop.encode('ascii')),
                             ctypes.byref(minimum), ctypes.byref(maximum))
        return minimum.value, maximum.value

    def _handleapi(self, res):
        if res:
            buffer = ctypes.create_string_buffer(256)
            self._api.getlasterror(self._device, buffer, ctypes.c_size_t(256))
            raise RuntimeError(
                'Basler Pylon Error: "{}"!'.format(buffer.value.decode('utf8')))

    def set(self, name, value):
        changed = []

        name = str(name).lower()
        self.check(name, 'w', value)

        if name in self._propertynamelut:
            prop = self._propertynamelut[name]
        else:
            prop = name
        changed = [name]

        with self._getsetRLock:

            if  name in ('controlframerate',):
                self._handleapi(
                    self._api.setbool(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.c_int(value)
                    )
                )

            elif name in ('triggersoftware', 'softwaresignalpulse'):
                self._api.commandExecute(
                    self._device,
                    ctypes.c_char_p(prop.encode('ascii')),
                )

            elif name in ('height', 'width', 'x', 'y',
                          'horizontalbinning', 'verticalbinning',
                          'burstframecount'):
                self._handleapi(
                    self._api.setint(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.c_longlong(value)
                    )
                )

            elif name in ('gain', 'exposuretime', 'offset',
                          'targetframerate', 'framerate',):
                if name == 'exposuretime':
                    value = value*1e6
                self._handleapi(
                    self._api.setfloat(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.c_double(value)
                    )
                )

            elif name in ('offsetselector', 'gainselector',
                          'gainauto',
                          'triggermode', 'triggersource',
                          'triggerselector', 'triggeractivation',
                          'exposuremode', 'exposureauto',
                          'softwaresignalselector',
                          'pixelformat', 'acquisitionmode',
                          'horizontalbinningmode', 'verticalbinningmode',
                          'iolineselector', 'iolinemode', 'iolinesource'):
                self._handleapi(
                    self._api.setenum(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.c_char_p(value.encode('ascii'))
                    )
                )

            elif name == 'averaging':
                acquiring = self.acquiring()
                self.pause()
                self._averaging = device.adjust(
                    value, self.property(prop).range)
                if acquiring:
                    self.resume()
                #self.acquire()

        return changed

    def get(self, name):
        result = None

        name = str(name).lower()
        self.check(name, 'r')

        if name in self._propertynamelut:
            prop = self._propertynamelut[name]
        else:
            prop = name

        with self._getsetRLock:

            if  name in ('controlframerate',):
                cint = ctypes.c_int(0)
                self._handleapi(
                    self._api.getbool(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.byref(cint)
                    )
                )
                result = bool(cint.value)

            elif name in ('lastframe', 'newframe', 'acquired'):
                with self._previewRLock:
                    result = self._preview.get(prop)

            elif name in ('triggersoftware', 'softwaresignalpulse'):
                cint = ctypes.c_int()
                self._api.commandDone(
                    self._device,
                    ctypes.c_char_p(prop.encode('ascii')),
                    ctypes.byref(cint)
                )
                result = bool(cint.value)

            elif name in ('height', 'width', 'x', 'y',
                          'horizontalbinning', 'verticalbinning',
                          'sensorwidth', 'sensorheight', 'burstframecount'):
                cint = ctypes.c_longlong(0)
                self._handleapi(
                    self._api.getint(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.byref(cint)
                    )
                )
                result = int(cint.value)

            elif name in ('gain', 'exposuretime', 'targetframerate',
                          'framerate', 'offset'):
                cdouble = ctypes.c_double(0.0)
                self._handleapi(
                    self._api.getfloat(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        ctypes.byref(cdouble)
                    )
                )
                result = float(cdouble.value)
                if name == 'exposuretime':
                    result *= 1e-6

            elif name == 'bitdepth':
                result = _pixelFormats[self.get('pixelformat')]

            elif name in ('offsetselector', 'gainselector',
                          'gainauto',
                          'triggermode', 'triggersource',
                          'triggerselector', 'triggeractivation',
                          'exposuremode', 'exposureauto',
                          'softwaresignalselector',
                          'pixelformat', 'acquisitionmode',
                          'horizontalbinningmode', 'verticalbinningmode',
                          'iolineselector', 'iolinemode', 'iolinesource'):
                cstr = ctypes.create_string_buffer(256)
                self._handleapi(
                    self._api.getenum(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        cstr,
                        ctypes.c_size_t(256)
                    )
                )
                result = cstr.value.decode('utf8')

            elif name in ('name', 'vendor', 'serial', 'version', 'firmware',):
                cstr = ctypes.create_string_buffer(256)
                self._handleapi(
                    self._api.getstr(
                        self._device,
                        ctypes.c_char_p(prop.encode('ascii')),
                        cstr,
                        ctypes.c_size_t(256)
                    )
                )
                result = cstr.value.decode('utf8')

            elif name == 'averaging':
                result = self._averaging

        return result

    def acquire(self, acqobj=None, resume=False):
        with self._getsetRLock:
            #if self.acquiring():
            self.stop(resumable=resume)

            self._allocframe()

            if self._averaging > 1:
                frametype = self._averageFrame.dtype
            else:
                frametype = self._frame.dtype

            self._ninaverage = 0
            self._averageFrame.fill(0)

            if not resume:
                self._acquisition = acqobj
                if acqobj is not None:
                    self._acquisition.init(self._frame.shape, frametype)
            self._preview.init(self._frame.shape, frametype)

            self._handleapi(
                self._api.acquire(
                    self._device,
                    ctypes.c_size_t(-1),
                    self._cAcquisitionCallback
                )
            )

    def resume(self):
        self.acquire(resume=True)

    def _acquisitionCallback(self, handle,
                             index, missed, buffer, width, height):
        # print(index, missed, buffer, width, height)
        try:
            if not self.connected():
                self.stop()
            else:
                frame = np.frombuffer(
                    (ctypes.c_byte*self._frame.nbytes).from_address(buffer),
                    dtype=self._frame.dtype)
                frame.shape = self._frame.shape
                if self._averaging > 1:
                    self._averageFrame += frame
                    self._ninaverage += 1
                    if self._ninaverage >= self._averaging:
                        self._averageFrame *= (1.0/self._ninaverage)

                        with self._previewRLock:
                            self._preview.appendone(self._averageFrame)
                        if self._acquisition is not None and \
                                    self._acquisition.appendone(
                                            self._averageFrame) <= 0:
                            #self.stop()
                            pass

                        self._ninaverage = 0
                        self._averageFrame.fill(0)
                else:
                    with self._previewRLock:
                        self._preview.appendone(frame)
                    if self._acquisition is not None and \
                            self._acquisition.appendone(frame) <= 0:
                        #self.stop()
                        pass

        except BaseException:
            self._acquisitionerror = sys.exc_info()
            self.stop()

    def acquiring(self):
        with self._getsetRLock:
            cint = ctypes.c_int(0)
            self._handleapi(self._api.isacquiring(
                self._device, ctypes.byref(cint)))
        return bool(cint.value)

    def stop(self, resumable=False):
        with self._getsetRLock:
            self._handleapi(self._api.stop(self._device))
            if not resumable:
                self._acquisition = None

    def pause(self):
        self.stop(resumable=True)

    def __repr__(self):
        return 'Pylon(cfg={})'.format(repr(self._cfg))

    def __str__(self):
        return self.__str__() + \
            ' # Pylon object at 0x{:>08X}.'.format(id(self))

    def __unicode__(self):
        return self.__str__()

    def __del__(self):
        # this can be unsafe
        try:
            self.stop()
            self._api.destroy(self._device)
        except:
            pass


# Create alias for find
find = Pylon.find

if __name__ == '__main__':
    dev = Pylon()
    dev.set('gain', 0.0)
    dev.set('exposuretime', 1e-3)
    acq = acquisition.Acquisition('inf')
    dev.acquire(acq)
