# -*- coding: utf-8 -*-

class Access(object):
    N = 0
    R = 1
    W = 2
    RW = 3
    MASK = 0x03
    def __init__(self, r=False, w=False):
        if isinstance(r, str):
            self._r = bool('r' in r)
            self._w = bool('w' in r)
        elif isinstance(r, int):
            if r & Access.R:
                self._r = True
            else:
                self._r = False
            if r & Access.W:
                self._w = True
            else:
                self._w = False
        elif isinstance(r, tuple):
            (self._r, self._w) = (bool(r[0]), bool(r[1]))
        else:
            self._r = bool(r)
            self._w = bool(w)

    def _get_r(self):
        return self._r
    def _set_r(self, value):
        self._r = bool(value)
    r = property(_get_r, _set_r, None, "Read access.")

    def _get_w(self):
        return self._w
    def _set_w(self, value):
        self._w = bool(value)
    w = property(_get_w, _set_w, None, "Write access.")

    def _get_rw(self):
        return self._r, self._w
    def _set_rw(self, rw):
        (self._w, self._r) = rw
    rw = property(_get_rw, _set_rw, None, "Read/write access.")

    def can(self, access):
        if not isinstance(access, Access):
            access = Access(access)
        if access.r and not self.r:
            return False
        if access.w and not self.w:
            return False
        return True

    def __str__(self):
        res = ''
        if self._r:
            res += 'r'
        if self._w:
            res += 'w'
        return 'Access(\'{}\')'.format(res)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__() + ' # Access object at 0x{:>08X}.'.format(id(self))


class Signal:

    # polarity
    HIGH = 0
    LOW = 1

    # edge
    RISING = 1
    FALLING = 2
    RISING_FALLING = 3
    BOTH = 3

    #conversion functions

    @staticmethod
    def Polarity(value):
        value = int(value)
        if value not in (Signal.HIGH, Signal.LOW):
            raise ValueError('Not a valid signal polarity.')
        return value

    @staticmethod
    def Edge(value):
        value = int(value)
        if value not in (Signal.RISING, Signal.FALLING, Signal.RISING_FALLING):
            raise ValueError('Not a valid signal edge.')
        return value

    @staticmethod
    def polarityStr(value):
        value = Signal.Polarity(value)
        if value == Signal.HIGH:
            return 'Signal.HIGH'
        else:
            return 'Signal.LOW'

    @staticmethod
    def edgeStr(value):
        value = Signal.Edge(value)
        if value == Signal.RISING:
            return 'Signal.RISING'
        elif value == Signal.FALLING:
            return 'Signal.FALLING'
        else:
            return 'Signal.RISING_FALLING'

class Property(object):
    def __init__(self, T=None, name="", description="",
                 access=Access(False, False),
                 size=0, default=None, range=None):
        self._type = T
        self._name = str(name)
        self._description = str(description)
        self._access = Access(access)
        self._size = int(size)
        self._default = default
        self._range = range

    def _get_type(self):
        return self._type
    type = property(_get_type, None, None, "Data type.")

    def _get_name(self):
        return self._name
    name = property(_get_name, None, None, "Property name.")

    def _get_description(self):
        return self._description
    description = property(_get_description, None, None, "Property description.")

    def _get_access(self):
        return self._access
    def _set_access(self, access):
        self._access = Access(access)
    access = property(_get_access, _set_access, None, "Current access flags.")

    def _get_size(self):
        return self._size
    size = property(_get_size, None, None, "Data field size in items.")

    def _get_default(self):
        return self._default
    def _set_default(self, default):
        self._default = default
    default = property(_get_default, _set_default, None, "Default value.")

    def _get_range(self):
        return self._range
    def _set_range(self, range):
        self._range = range
    range = property(_get_range, _set_range, None, "Value range.")

    def __str__(self):
        typename = None
        if hasattr(self._type, '__name__'):
            typename = self._type.__name__
        return 'Property(T={}, name=\'{}\', description=\'{}\', ' \
            'access={}, size={}, default={}, range={})'. \
            format(typename, self._name, self._description, \
            self._access, self._size, self._default, self._range)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__() + ' # Property object at 0x{:>08X}.'.format(id(self))


def is_within(value, r):
    if isinstance(r, Property):
        r = r.range

    if isinstance(value, str):
        if value not in r:
            return False
        else:
            return True
    elif isinstance(value, Roi):
        return r >= value
    elif isinstance(value, int) or isinstance(value, float):
        if r is not None:
            for i in range(0, len(r), 2):
                if value >= r[i] and value <= r[i + 1]:
                    return True
            return False
        else:
            return True

def adjust(value, r):
    if isinstance(r, Property):
        r = r.range

    if r is not None:
        distance = float('inf')
        ind = -1
        for i in range(0, len(r), 2):
            if value >= r[i] and value <= r[i + 1]:
                ind = -1
                break
            d = min(abs(r[i] - value), abs(r[i + 1] - value))
            if d < distance:
                distance = d
                ind = i

        if ind >= 0:
            if value < r[ind]:
                value = r[ind]
            else:
                value = r[ind + 1]

    return value


class Descriptor(object):

    @staticmethod
    def fromDict(value):
        return Descriptor(
            t=value.get('prettyName', ''),
            name=value.get('name', ''),
            manufacturer=value.get('manufacturer', ''),
            serial=value.get('serial', ''),
            port=value.get('port', ''),
        )

    def __init__(self, prettyname="", name="", manufacturer="", serial="",
                 port=None, **kwargs):

        self._prettyname = str(prettyname)
        self._name = str(name)
        self._manufacturer = str(manufacturer)
        self._serial = str(serial)
        self._port = port

        self._others = kwargs

    def _get_prettyname(self):
        return self._prettyname
    def _set_prettyname(self, value):
        self._prettyname = str(value)
    prettyName = property(_get_prettyname, _set_prettyname, None,
                          "Short device name.")

    def _get_name(self):
        return self._name
    def _set_name(self, value):
        self._name = str(value)
    name = property(_get_name, None, None, "Long device name.")

    def _get_manufacturer(self):
        return self._manufacturer
    def _set_manufacturer(self, value):
        self._manufacturer = str(value)
    manufacturer = property(_get_manufacturer, _set_manufacturer, None,
                            "Device manufacturer.")

    def _get_serial(self):
        return self._serial
    def _set_serial(self, value):
        self._serial = str(value)
    serial = property(_get_serial, _set_serial, None, "Device serial.")

    def _get_port(self):
        return self._port
    def _set_port(self, value):
        self._port = value
    port = property(_get_port, _set_port, None, "Device port.")

    def _get_others(self):
        return self._others
    others = property(_get_others, None, None, "Other descriptors.")

    def asDict(self):
        return {'prettyName':self.prettyName, 'name':self.name,
                'manufacturer':self.manufacturer, 'serial':self.serial,
                'port':self.port, 'others':self.others}

    def __str__(self):
        return 'Descriptor(prettyname=\'{}\', name=\'{}\', manufacturer=\'{}\', ' \
                'serial=\'{}\', port={}, **{})'. \
            format(self._prettyname, self._name, self._manufacturer, \
                    self._serial, self._port, self._others)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__() + \
            ' # Descriptor object at 0x{:>08X}.'.format(id(self))

class Device(object):
    def __init__(self, descriptor):
        if not isinstance(descriptor, Descriptor):
            raise TypeError("Requires Descriptor object input argument.")
        self._descriptor = descriptor
        self._properties = {}

    def descriptor(self):
        return self._descriptor

    def set(self, name, value):
        return ()

    def get(self, name):
        return None

    def addProperty(self, prop):
        if not isinstance(prop, Property):
            raise TypeError("Requires Property object input argument.")
        self._properties[prop.name] = prop

    def addProperties(self, props):
        for prop in props:
            self.addProperty(prop)

    def isProperty(self, name):
        return name in self._properties

    def property(self, name):
        return self._properties.get(name)

    def properties(self):
        return self._properties

    def check(self, name, access, value=None):
        name = str(name)
        if not name in self._properties:
            raise NameError('Property \'{}\' is not defined'.format(name))
        prop = self._properties[name]
        if not prop.access.can(access):
            raise AttributeError('Property \'{}\' is inaccessible.'.\
                format(name))
        if value is not None:
            # num_items = 1
            iterable = hasattr(value, '__len__') and not isinstance(value, str)
            # if iterable:
            #     num_items = len(value)
            # if num_items > prop.size:
            #    raise ValueError(
            #         'The number of items in the given value ({:d}) exceeds '\
            #        'the maximum size ({:d}) of property \'{:s}\'!'.format(
            #             len(value), prop.size, name))
            if prop.size > 1 and iterable:
                for index, item in enumerate(value):
                    if not isinstance(item, prop.type):
                        raise ValueError(
                            'Value type \'{}\' at offset {:d} incompatible '\
                            'with the \'{}\' property type \'{}\'.'.format(
                                type(item).__name__, index,
                                name, prop.type.__name__)
                        )
            elif not isinstance(value, prop.type):
                raise ValueError(
                    'Value type \'{}\' incompatible with the '
                    '\'{}\' property type \'{}\'.'.format(
                        type(value).__name__, name, prop.type.__name__)
                )

    def info(self):
        out = 'Device supports the following properties:\n'
        for prop in self._properties:
            out += '\t'
            out += str(self._properties[prop])
            out += '\n'
        return out

class Roi(object):
    def __init__(self, width, height=1, x=0, y=0):

        if isinstance(width, Roi):
            roi = width
            width = roi._width
            height = roi._height
            x = roi._x
            y = roi._y
        elif isinstance(width, list) or isinstance(width, tuple):
            roi = width
            width = roi[0]
            height = roi[1]
            x = y = 0
            if len(roi) > 2:
                x = roi[2]
                if len(roi) > 3:
                    y = roi[3]

        self._width = int(width)
        self._height = int(height)
        self._x = int(x)
        self._y = int(y)

    def _get_size(self):
        return self._width*self._height
    size = property(_get_size, None, None, "Window size in pixels.")

    def _get_width(self):
        return self._width
    def _set_width(self, value):
        self._width = int(value)
    width = property(_get_width, _set_width, None, "Window width.")

    def _get_height(self):
        return self._height
    def _set_height(self, value):
        self._height = int(value)
    height = property(_get_height, _set_height, None, "Window height.")

    def _get_x(self):
        return self._x
    def _set_x(self, value):
        self._x = int(value)
    x = property(_get_x, _set_x, None, "Window horizontal offset.")

    def _get_y(self):
        return self._y
    def _set_y(self, value):
        self._y = int(value)
    y = property(_get_y, _set_y, None, "Window vertical offset.")

    def adjust(self, maxroi):
        if self.x < maxroi.x:
            self.x = maxroi.x
        elif self.x > maxroi.x + maxroi.width:
            self.x = maxroi.x + maxroi.width

        if self.y < maxroi.y:
            self.y = maxroi.y
        elif self.y > maxroi.y + maxroi.height:
            self.y = maxroi.y + maxroi.height

        if self.x + self.width > maxroi.x + maxroi.width:
            self.width = maxroi.x + maxroi.width - self.x

        if self.y + self.height > maxroi.y + maxroi.height:
            self.height = maxroi.y + maxroi.height - self.y

        return self

    def __eq__(self, other):
        if not isinstance(other, Roi):
            raise ValueError("Requires Roi object input argument.")
        return self.width == other.width and \
                self.height == other.height and \
                self.x == other.x and self.y == other.y

    def __ge__(self, other):
        if not isinstance(other, Roi):
            raise ValueError("Requires Roi object input argument.")
        return (self.x <= other.x) and (self.y <= other.y) and \
                (self.x + self.width >= other.x + other.width) and \
                (self.y + self.height >= other.y + other.height)

    def __gt__(self, other):
        if not isinstance(other, Roi):
            raise ValueError("Requires Roi object input argument.")
        return self.width*self.height > other.width*other.height and \
                self.x <= other.x and self.y <= other.y and \
                self.x + self.width >= other.x + other.width and \
                self.y + self.height >= other.y + other.height

    def __le__(self, other):
        if not isinstance(other, Roi):
            raise ValueError("Requires Roi object input argument.")
        return self.x >= other.x and self.y >= other.y and \
                self.x + self.width <= other.x + other.width and \
                self.y + self.height <= other.y + other.height

    def __lt__(self, other):
        if not isinstance(other, Roi):
            raise ValueError("Requires Roi object input argument.")
        return self.width*self.height < other.width*other.height and \
                self.x > other.x and self.y > other.y and \
                self.x + self.width <= other.x + other.width and \
                self.y + self.height <= other.y + other.height

    def __str__(self):
        return 'Roi(width={}, height={}, x={}, y={})'. \
            format(self.width, self.height, self.x, self.y)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__() + ' # Roi object at 0x{:>08X}.'.format(id(self))
