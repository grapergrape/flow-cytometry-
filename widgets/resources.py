# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 05:22:14 2017

@author: miran
"""
import os.path

from PyQt5.QtGui import QIcon, QPixmap
from widgets import ICON_PATH

CACHED_ICONS = {}
CACHED_PIXMAPS = {}

def loadIcon(filename):
    if CACHED_ICONS:
        return CACHED_ICONS[filename]
    else:
        return QIcon(
            os.path.join(ICON_PATH, filename))

def loadPixmap(filename):
    if CACHED_PIXMAPS:
        return CACHED_PIXMAPS[filename]
    else:
        return QPixmap(
            os.path.join(ICON_PATH, filename))

def loadResources():
    import os.path
    import os
    fileNames = os.listdir(ICON_PATH)

    icons = {}
    pixmaps = {}
    for fileName in fileNames:
        path, file_ext = os.path.split(fileName)
        filename, ext = os.path.splitext(file_ext)
        ext = ext.lower()
        if ext == '.png':
            icons[file_ext] = loadIcon(fileName)
            pixmaps[file_ext] = loadPixmap(fileName)

    global CACHED_ICONS, CACHED_PIXMAPS
    CACHED_ICONS = icons
    CACHED_PIXMAPS = pixmaps

    return icons, pixmaps
