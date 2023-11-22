# -*- coding: utf-8 -*-
"""
Created on Wen Feb 21 22:51:01 2017

@author: miran
"""
import pickle
import os.path

import numpy as np
import scipy.io
import h5py

def h5SaveRecursive(h5file, **kwargs):
    '''
    Save data to a h5file. The datasets are saved using the argument names.
    The save process is recursive for dict values.

    Parameters
    ----------
    h5file: h5py.File
        A HDF5 file.
    kwargs: dict
        A dict of keyword arguments
    '''
    for field, value in kwargs.items():
        if isinstance(value, dict):
            h5SaveRecursive(h5file.create_group(field), **value)
        else:
            if value is None:
                value = []
            h5file[field] = value

def h5LoadRecursive(h5file):
    '''
    Load data from a HDF5 file into a dict from a file that was saved by
    h5SaveRecursive.

    Parameters
    ----------
    h5file: h5py.File
        A HDF5 file.

    Returns
    -------
    data: dict
        Loaded data.
    '''
    result = {}
    for key in h5file.keys():
        value = h5file[key]
        if isinstance(value, h5py.Group):
            result[key] = h5LoadRecursive(value)
        else:
            result[key] = value[()]

    return result

def exportData(file, format=None, options=None, **kwargs):
    '''
    Export data to a desired format. Data can be exported to:
        - numpy .npz file (compressed numpy files with multiple arrays),
        - matlab .mat file,
        - HDF5 .h5 file,
        - python .pkl file.

    Parameters
    ----------
    file: str ot file like object
        Output file or file name.
    format: str
        File format. If None (default), it is deduced from the file name.
    opt: dict
        Keyword arguments for the exporter.
    kwargs: dict
        Data to be saved.
    '''
    filename = fileobj = None

    if options is None:
        options = {}

    if isinstance(file, str):
        filename = file
    else:
        fileobj = file
        if hasattr(fileobj, 'name'):
            filename = file.name

    if format is None:
        if filename is not None and filename:
            ext = os.path.splitext(filename)[1]
            format = ext

    if format is None:
        raise ValueError(
            'Export file format is not specified and cannot be deduced '
            'from the file name!'
        )

    if format in ('npz', '.npz'):
        if fileobj is None:
            with open(filename, 'wb') as fileobj:
                np.savez_compressed(fileobj, **kwargs)
        else:
            np.savez_compressed(fileobj, **kwargs)

    elif format in ('mat', '.mat'):
        opts = options.get('mat', {})
        if fileobj is None:
            with open(filename, 'wb') as fileobj:
                scipy.io.savemat(fileobj, kwargs, **opts)
        else:
            scipy.io.savemat(fileobj, kwargs, **opts)

    elif format in ('h5', '.h5'):
        opts = options.get('h5', {})
        if fileobj is None:
            with h5py.File(filename, 'w', **opts) as fileobj:
                h5SaveRecursive(fileobj, **kwargs)
        else:
            h5SaveRecursive(fileobj, **kwargs)

    elif format in ('pkl', '.pkl', 'pickle', '.pickle'):
        opts = options.get('pkl', {})
        if fileobj is None:
            with open(filename, 'wb') as fileobj:
                pickle.dump(kwargs, fileobj, **opts)
        else:
            pickle.dump(kwargs, fileobj, **opts)

    else:
        raise ValueError('File format "{}" not supported!'.format(format))

    return True
