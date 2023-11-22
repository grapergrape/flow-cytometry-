# -*- coding: utf-8 -*-

import time
import threading
import sys
import pickle
import os.path
import struct
import io
import glob

import numpy

try:
    import cv2
except ImportError:
    print("OpenCV library not available. Recording raw AVI will not be possible!")

from common import device

PICKLE_PROTOCOL = 3
PROTOCOL_VERSION = 1.0

class Exporter(object):

    def __init__(self, ofile, splitsize=1073741824, header=None,
                 buffering=16000000, overwrite=True):
        owns_file = False

        if isinstance(ofile, Exporter):
            # initialize from exporter
            exporterObj = ofile
            self._buffering = exporterObj.buffering
            self._splitSize = exporterObj.splitsize
            self._ofile = exporterObj.file
            self._splitCounter = exporterObj.splits
            self._owns_file = exporterObj.owner
            self._offsets = []
            self._userHeader = exporterObj.header
            self._protocolHeader = exporterObj.protocol

            self._fileName, self._fileExt, counter = \
                self._prepareFilename(self._ofile)
        else:
            self._buffering = int(buffering)
            if isinstance(ofile, str):
                ofile = os.path.abspath(ofile)
                if os.path.exists(ofile) and not overwrite:
                    raise IOError('File allready exists.')
                ofile = open(ofile, 'wb', buffering=self._buffering)
                owns_file = True
            elif isinstance(ofile, io.IOBase):
                if 'b' not in ofile.mode or 'w' not in ofile.mode:
                    raise ValueError('File object must be binary and writable.')
            else:
                raise ValueError('A valid file object is required.')

            self._fileName, self._fileExt, self._splitCounter = \
                self._prepareFilename(ofile)
            self._ofile = ofile
            self._splitSize = max(int(splitsize), 0)
            self._offsets = []
            self._buffering = int(buffering)
            self._owns_file = owns_file

            # create and write exporter header
            self._protocolHeader = {
                'type':type(self),
                'version':PROTOCOL_VERSION,
                'buffeing':buffering,
                'splitsize':splitsize,
                'file':self._fileName,
                'epochtime':time.time(),
                'localtime':time.localtime()
            }

            pickle.dump(self._protocolHeader, self._ofile, PICKLE_PROTOCOL)

            # write the header
            if header is None:
                header = {}
            self._userHeader = header
            pickle.dump(header, self._ofile, PICKLE_PROTOCOL)

    def _prepareFilename(self, ofile):
        splitCounter = 0
        # parse the file name for number ('blabla_number')
        name, ext = os.path.splitext(ofile.name)
        res = name.rsplit('_')
        base = res[0]
        if len(res) > 1:
            counterStr = res[1]
        else:
            counterStr = ''
        if counterStr:
            try:
                splitCounter = int(counterStr)
            except ValueError:
                pass
        return base, ext, splitCounter

    def _get_buffering(self):
        return self._buffering
    buffering = property(_get_buffering, None, None,
                         'File buffering size (byte).')

    def _get_splitsize(self):
        return self._splitSize
    splitsize = property(_get_splitsize, None, None, 'File split size (byte).')

    def _get_file(self):
        return self._ofile
    file = property(_get_file, None, None, 'OS file object.')

    def _get_splits(self):
        return self._splitCounter
    splits = property(_get_splits, None, None, 'Number of file splits.')

    def _get_file_owner(self):
        return self._owns_file
    owner = property(
        _get_file_owner, None, None,
        'True if this Exporter instance is the OS file object owner.'
    )

    def _get_header(self):
        return self._userHeader
    header = property(_get_header, None, None, 'User file header object.')

    def _get_protocol_header(self):
        return self._protocolHeader
    protocol = property(_get_protocol_header, None, None,
                        'Protocol header/descriptor.')

    def write(self, item):
        pos = self._ofile.tell()
        if pos >= self._splitSize and self._splitSize > 0:
            # save footer to the file amd close if owner
            self.close()
            # generate a file name
            nextFileName = '{}_{}'.format(self._fileName, self._splitCounter)
            if self._fileExt:
                nextFileName += self._fileExt
            self._splitCounter += 1
            # open a new file
            self._ofile = open(nextFileName, 'bw', self._buffering)
            self._owns_file = True
            # write exporter protocol header
            pickle.dump(self._protocolHeader, self._ofile, PICKLE_PROTOCOL)
            # write user header
            pickle.dump(self._userHeader, self._ofile, PICKLE_PROTOCOL)
            # initialize footer data ... offsets of items
            self._offsets = []

        self._offsets.append(self._ofile.tell())
        pickle.dump(item, self._ofile, PICKLE_PROTOCOL)

    def _writefooter(self):
        # get the start offset of the footer
        pos = self._ofile.tell()
        # write footer to the file
        pickle.dump({'offsets':self._offsets}, self._ofile, PICKLE_PROTOCOL)
        # finish the file with offset of the footer
        self._ofile.write(struct.pack('<Q', pos))

    def flush(self):
        self._ofile.flush()

    def close(self):
        self._writefooter()
        if self._owns_file:
            self._ofile.close()
            self._offsets = []

    def __del__(self):
        # this might be dangerous
        try:
            self.close()
        except (ValueError,):
            pass

class Importer:
    def __init__(self, ifile):
        owns_file = False

        if isinstance(ifile, str):
            ifile = os.path.abspath(ifile)
            if not os.path.exists(ifile):
                raise IOError('File does not exist.')
            ifile = open(ifile, 'rb')
            owns_file = True
        elif isinstance(ifile, io.IOBase):
            if 'b' not in ifile.mode or 'r' not in ifile.mode:
                raise ValueError('File object must be binary and readable.')
        else:
            raise ValueError('A valid file object is required.')

        ifileNames = [ifile.name]
        # read additional files
        ifileNames += glob.glob('{}_[0-9]*'.format(ifileNames[0]))
        name, ext = os.path.splitext(ifileNames[0])
        ifileNames += glob.glob('{}_[0-9]*{}'.format(name, ext))

        # read exporter header - file information
        ifile.seek(0, os.SEEK_SET)
        self._protocolHeader = pickle.load(ifile)

        # user header offset
        self._userHeaderOffset = ifile.tell()

        self._ifileNames = tuple(ifileNames)
        self._ifiles = [ifile]
        self._num_items = []
        self._file_indices = []
        self._file_offsets = []
        self._owns_file_0 = owns_file

        for index, ifileName in enumerate(self._ifileNames):
            if index > 0:
                ifile = open(ifileName, 'rb')
                self._ifiles.append(ifile)
            ifile = self._ifiles[-1]
            ifile_offsets = self._get_offsets(ifile)
            self._num_items.append(len(ifile_offsets))

            self._file_indices += [index]*self._num_items[-1]
            self._file_offsets += ifile_offsets

        self._ifiles = tuple(self._ifiles)
        self._num_items = tuple(self._num_items)
        self._file_indices = tuple(self._file_indices)
        self._file_offsets = tuple(self._file_offsets)

    def _get_offsets(self, ifile):
        ifile.seek(-8, os.SEEK_END)
        where, = struct.unpack('<Q', ifile.read(8))
        ifile.seek(where)
        return pickle.load(ifile)['offsets']

    def _get_size(self):
        return len(self._file_indices)
    size = property(_get_size, None, None, 'Total number of items.')

    def item(self, item_index):
        if item_index < 0:
            item_index += self.size
        if item_index < 0 or item_index >= self.size:
            raise IndexError('Item index out of range.')

        file_index = self._file_indices[item_index]
        file_offset = self._file_offsets[item_index]
        return self._item(file_index, file_offset)

    def _get_header(self):
        self._ifiles[0].seek(self._userHeaderOffset)
        return pickle.load(self._ifiles[0])
    header = property(_get_header, None, None, 'User file header object.')

    def _get_protocol_header(self):
        return self._protocolHeader
    protocol = property(_get_protocol_header, None, None,
                        'Protocol header/descriptor.')

    def _get_file_names(self):
        return self._ifileNames
    filenames = property(_get_file_names, None, None,
                         'Imported files (names).')

    def _get_files(self):
        return self._ifiles
    files = property(_get_files, None, None,
                     'Imported file objects.')

    def _item(self, file_index, file_offset):
        self._ifiles[file_index].seek(file_offset)
        #print(index + 1)
        return pickle.load(self._ifiles[file_index])

    def __len__(self):
        return self.size

    def __getitem__(self, item_index):
        if isinstance(item_index, slice):
            return [self.item(i)
                    for i in range(*item_index.indices(len(self)))]
        else:
            return self.item(item_index)

    def close(self):
        for index, ifile in enumerate(self._ifiles):
            if self._owns_file_0 or index > 0:
                if ifile:
                    ifile.close()

    def __iter__(self):
        return (self.item(index) for index in range(self.size))

    def __del__(self):
        self.close()


class AviExporter(Exporter):
    def __init__(self, fileobj: str, fps: float,
                 overwrite: bool = True, **kwargs):
        #super().__init__()
        self._writer = None
        self._fps = fps
        self._overwrite = overwrite
        self._frame_buffer = None
        self._splitSize = 0
        self._owns_file = True
        self._buffering = None
        self._ofile = None
        if not isinstance(fileobj, str):
            raise TypeError('File object must be a string path!')
        if os.path.isfile(fileobj) and not overwrite:
            raise FileExistsError('Cannot overwrite an existing Video '
                                    'file "{}"!'.format(fileobj))

        self._ofile = fileobj

    def write(self, item):
        # ignore header data ... cannot write to AVI file
        if isinstance(item, dict):
            return

        # this should be  frame
        frame, ts = item
        
        frame = numpy.asarray(frame, dtype=numpy.uint8)
        if frame.dtype != numpy.uint8:
            raise TypeError(
                'Only 8-bit unsigned integer type frames are supported!')

        if self._writer is None:
            self._height, self._width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self._ofile, 0, self._fps, (self._width, self._height))
            self._frame_buffer = numpy.zeros(
                (self._height, self._width, 3), dtype=numpy.uint8)
        elif frame.shape[:2] != (self._height, self._width):
            raise ValueError('The frame size has changed!')

        if frame.ndim == 2:
            frame = numpy.stack([frame]*3, axis=2, out=self._frame_buffer)
        else:
            raise TypeError('Only 2D grayscale or 3D RGB images are supported!')

        self._writer.write(frame)

    def _get_buffering(self):
        return self._buffering
    buffering = property(_get_buffering, None, None,
                         'File buffering size (byte).')

    def _get_splitsize(self):
        return self._splitSize
    splitsize = property(_get_splitsize, None, None, 'File split size (byte).')

    def _get_file(self):
        return self._ofile
    file = property(_get_file, None, None, 'OS file object.')

    def _get_splits(self):
        return 0
    splits = property(_get_splits, None, None, 'Does not support video splitting.')

    def _get_file_owner(self):
        return self._owns_file
    owner = property(
        _get_file_owner, None, None,
        'True if this Exporter instance is the OS file object owner.'
    )

    def _get_header(self):
        return None
    header = property(_get_header, None, None, 'Does not support headers.')

    def _get_protocol_header(self):
        return None
    protocol = property(_get_protocol_header, None, None,
                        'Does not support protocol header.')

    def flush(self):
        self._ofile.flush()

    def close(self):
        if self._owns_file:
            if self._writer is not None:
                self._writer.release()

    def __del__(self):
        # this might be dangerous
        try:
            self.close()
        except (ValueError,):
            pass


class Acquisition(device.Device):
    INF = sys.maxsize

    def __init__(self, n=1, ofile=None, callback=None, header=None,
                 buffering=16000000, splitsize=1073741824):

        if n == 'inf':
            n = Acquisition.INF

        n = max(int(n), 0)
        if n == Acquisition.INF:
            nstr = 'inf'
        else:
            nstr = str(n)

        if ofile is not None:
            if isinstance(ofile, str) or isinstance(ofile, io.IOBase):
                ofile = Exporter(ofile, splitsize=splitsize,
                                 header=header, buffering=buffering)
            elif isinstance(ofile, Exporter):
                pass
            else:
                raise ValueError('Input argument ofile should be a valid file, '\
                                 'string or Exporter instance.')

        self._ofile = ofile
        device.Device.__init__(
            self, device.Descriptor('Acquisition',
                                    'Acquisition object {}'.format(nstr),
                                    '', '', -1))

        self._callback = callback

        self._data = None
        self._timestamp = None
        self._itemshape = None
        self._metadata = None

        self._nacquired = 0
        self._n = n
        self._lastget = 0

        self._getsetlock = threading.Lock()
        self._endevent = threading.Event()

        self.addProperties(
            (
                device.Property(int, 'n',
                                'Number of items to acquire.',
                                'rw', 1, 0, (0, Acquisition.INF)),
                device.Property(None, 'close',
                                'Close the file io object if any.',
                                'w', 0),
                device.Property(None, 'stop',
                                'Stop the acquisition.',
                                'w', 0),
                device.Property(int, 'acquired',
                                'Number of acquired items.',
                                'r', 1),
                device.Property(numpy.ndarray, 'last',
                                'Last acquired item or None.',
                                'r', 1),
                device.Property(numpy.ndarray, 'new',
                                'New acquired item or None.',
                                'r', 1),
                device.Property(dict, 'metadata',
                                'Device dependent metadata dictionary.',
                                'r', 1),
                device.Property(numpy.ndarray, 'data',
                                'All acquired data as a numpy array.',
                                'r', 1),
                device.Property(numpy.ndarray, 'timestamp',
                                'Timestamps of acquired data as a numpy array.',
                                'r', 1),
            )
        )

    def init(self, itemshape, datatype, metadata=None):
        self._itemshape = tuple(itemshape)
        self._metadata = metadata
        if self._n == Acquisition.INF or self._ofile is not None:
            n = 1
        else:
            n = self._n
        self._lastget = self._nacquired
        shape = (n,) + self._itemshape
        self._data = numpy.zeros(shape, dtype=datatype)
        self._timestamp = numpy.zeros([n], dtype='float64')
        if self._ofile is not None:
            if metadata is None:
                metadata = {}
            self._ofile.write({'metadata':metadata})

    def appendone(self, item, timestamp=None):
        result = 0
        if timestamp is None:
            timestamp = time.time()

        with self._getsetlock:
            if self._n > self._nacquired or self._n == Acquisition.INF:
            # if not self._endevent.is_set(): # is much slower
                if self._itemshape == item.shape and \
                        self._data.dtype.type.__name__ == \
                        item.dtype.type.__name__:

                    if self._callback is not None:
                        self._callback.process(self, item, timestamp,
                                               self._nacquired + 1)

                    if self._n == Acquisition.INF:
                        ind = 0
                        result = Acquisition.INF
                    else:
                        if self._ofile is not None:
                            ind = 0
                        else:
                            ind = self._nacquired
                        result = self._n - self._nacquired - 1

                    self._data[ind] = item
                    self._timestamp[ind] = timestamp

                    # write data to file if required
                    if self._ofile is not None:
                        self._ofile.write((item, timestamp))

                    self._nacquired += 1

        if result <= 0:
            self._endevent.set()

        return result

    def appendmany(self, items, timestamp=None):
        result = 0
        if timestamp is None:
            timestamp = time.time()

        nitems = len(items)

        with self._getsetlock:
            if self._itemshape == items[0].shape and \
                    self._data.dtype.type.__name__ == \
                    items[0].dtype.type.__name__:

                if self._n == Acquisition.INF:
                    # save only the last sample
                    self._data[0] = items[-1]
                    # save to file all samples
                    if self._ofile is not None:
                        self._ofile.write((items, timestamp))

                    result = Acquisition.INF

                elif self._n > self._nacquired:
                    # the number of items that still has to be acquired
                    nitems = min(self._n - self._nacquired, nitems)
                    self._data[self._nacquired: self._nacquired + nitems] = \
                        items[:nitems]

                    # save to file only the required samples samples
                    if self._ofile is not None:
                        self._ofile.write((items[:nitems], timestamp))

                    result = self._n - self._nacquired - nitems

                self._nacquired += nitems

        if result <= 0:
            self._endevent.set()

        return result

    def get(self, name):
        value = None

        with self._getsetlock:

            self.check(name, 'r')

            if name == 'n':
                value = self._n
            elif name == 'acquired':
                value = self._nacquired
            elif name == 'last':
                if self._nacquired > 0:
                    if self._n == Acquisition.INF or self._ofile is not None:
                        ind = 0
                        self._lastget = self._nacquired
                    else:
                        self._lastget = self._nacquired
                        ind = self._lastget - 1
                    value = numpy.array(self._data[ind])

            elif name == 'new':
                if self._lastget < self._nacquired:
                    if self._n == Acquisition.INF or self._ofile is not None:
                        ind = 0
                        self._lastget = self._nacquired
                    else:
                        self._lastget = self._nacquired
                        ind = self._lastget - 1
                    value = numpy.array(self._data[ind])
            elif name == 'metadata':
                value = self._metadata
            elif name == 'data':
                value = self._data
            elif name == 'timestamp':
                value = self._timestamp

        return value

    def set(self, name, value=None):
        changed = []

        self.check(name, 'w', value)

        with self._getsetlock:

            if name == 'n':
                n = max(self._nacquired, value)
                self._n = n
                if self._n <= self._nacquired:
                    self._endevent.set()
                elif self._n > self._nacquired:
                    self._endevent.clear()
                changed = [name]

            elif name == 'stop':
                self._endevent.set()
                self._n = self._nacquired
            elif name == 'close':
                self._ofile.close()
                self._ofile = None

        return changed

    def stop(self):
        self.set('stop')

    def join(self, timeout=None):
        return self._endevent.wait(timeout)

    def export(self, exporter):
        # export the acquired data by using the specified exporter
        n = int(self._data.size/numpy.prod(self._itemshape))
        for ind in range(n):
            item = self._data[ind]
            item.shape = self._itemshape
            exporter.write(item)

if __name__ == '__main__':
    '''
    def test(n, shape, T):
        a = Acquisition(n)
        a.init(shape, T)
        data = numpy.zeros(shape, dtype=T)
        t0 = time.time()
        for i in range(n):
            a.appendone(data)
        return time.time() - t0, a.get('acquired')
    '''
    '''
    def testw(f, size=[320, 256], n=100, buffer=None):
        import numpy as np
        import pickle
        import time
        import acquisition

        T = [0.0]*3
        dsize = [n]
        dsize.extend(size)
        data = np.random.randint(0, 65535, dsize).astype(np.uint16)
        out = np.zeros_like(data)
        acq = acquisition.Acquisition(
            n, acquisition.Exporter(f + '1', buffering=buffer))
        acq.init([320, 256], 'uint16')

        if buffer:
            fid = open(f, 'wb', buffer)
        else:
            fid = open(f, 'wb')
        t1 = time.perf_counter()
        pickle.dump(data, fid, -1)
        T[0] = time.perf_counter() - t1
        fid.close()

        t1 = time.perf_counter()
        for i in range(n):
            out[i, :, :] = data[i]
        T[1] = time.perf_counter() - t1

        t1 = time.perf_counter()
        for i in range(n):
            acq.appendone(data[i])
        T[2] = time.perf_counter() - t1
        del acq
        return
        '''
    import time
    from pybox.simcam.simcam import Simcam
    cfg = {'sensorwidth': 640, 'sensorheight': 480, 'bitdepth': 8,
           'gaintype': float, 'gainrange': (1.0, 16.0),}

    #frames = numpy.random.randint(0, 255, (100, 1080, 1920), dtype=numpy.uint8)
    #exporter = AviExporter("test.avi", fps=25)
    #acq = Acquisition(n=frames.shape[0], ofile=exporter)
    #acq.init(frames.shape[1:], frames.dtype)
    #t_start = time.perf_counter()
    #for frame in frames:
    #    acq.appendone(frame)
    #dt = time.perf_counter() - t_start
    #acq.join()
    #print(dt)

    #exit(0)

    acq = Acquisition(n=100, ofile=AviExporter("test.avi", fps=25))
    cam = Simcam(cfg)
    cam.set('exposuretime', 100e-3)
    cam.set('gain', 4.)

    cam.acquire(acq)
    acq.join()

