# -*- coding: utf-8 -*-
from widgets import common, camera

if __name__  == '__main__':
    import argparse
    from widgets import common

    parser = argparse.ArgumentParser(description='Spectrometer viewer')
    parser.add_argument('-d', '--device', metavar='DEVICE',
                        type=str, default='sim', choices=['sim', 'basler',
                                                          'xeva', 'mct', 'andor'],
                        help='Camera device type')
    parser.add_argument('-s', '--serial', metavar='SERIAL_NUMBER',
                        type=str, default='',
                        help='Camera serial number')
    parser.add_argument('-l', '--listdevices', action="store_true",
                        help='List cameras - supported for Basler devices')

    args = parser.parse_args()
    device = args.device
    serial = args.serial
    listdevices = args.listdevices

    app = common.prepareqt()

    camera_device = None
    title = 'Camera viewer'
    uicfg = None

    from basler import pylon
    if listdevices:
        devices = pylon.Pylon.find()
        print('\n' + '='*80)
        print('Found {:d} Basler devices!'.format(len(devices)))
        print('='*80)
        for index, device in enumerate(devices):
            if index != 0 and index < len(devices):
                print('-'*80)
            print('{}. Serial:"{}"'.format(index + 1, device['serial']))
            for property, value in device.items():
                print('    {}: {}'.format(property, value))
        print('='*80)

    else:
        camera_device = pylon.Pylon(pylon.Configuration(serial))
        if camera_device is not None:
            descriptor = camera_device.descriptor()
            title = 'Basler {:s} "{:s}" viewer'.format(
                descriptor.prettyName, descriptor.serial)

    if camera_device is not None:
        # print some basic information
        # print(title, camera.descriptor())
        camera_device.set('exposureauto', 'Off')
        camera_device.set('gainauto', 'Off')
        camera_device.set('exposuretime', 0.001)
        
        # creating camera hub
        hub = camera.CameraHub(camera_device, cfg=uicfg)
        hub.setWindowTitle(title)

        # adding a basic image processing and acquisition module
        sdm = camera.DataHubModule()
        hub.connectModule(sdm)

        hub.show()

        app.exec_()
