# -*- coding: utf-8 -*-
import os.path
import traceback
from time import sleep

import numpy as np
from scipy.signal import windows
from scipy.interpolate import interp1d
from skimage.restoration import unwrap_phase
from skimage.transform import resize

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, \
                            QHBoxLayout, QComboBox, QListView, \
                            QDialog, QLineEdit, QFileDialog, QMessageBox, \
                            QCheckBox, QGridLayout, QProgressBar, QSlider, QSizePolicy
from PyQt5.QtGui import QPixmap, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt, QCoreApplication, QSize, QThread, QObject
from PyQt5.QtCore import pyqtSignal as QSignal

from widgets import imview, common, dialogs, resources, fpa
from common import misc, export, acquisition

PICKLE_PROTOCOL = 3

DEFAULT_CONFIGURATION = fpa.DEFAULT_CONFIGURATION

def clip(value, low, high):
    return max(min(value, high), low)


class ImageViewDialog(QDialog):
    def __init__(self, parent: object = None):
        '''
        Creates a dialog window for showing images.

        Parameters
        ----------
        parent: object
            Parent window/widget.
        '''
        super().__init__(parent, Qt.Dialog)
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)

        self.previewWidget = imview.Image()
        self.previewWidget.hideProfiles()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.previewWidget)
        self.setLayout(layout)

        self._position = self._size = None

    def setImage(self, data: np.ndarray):
        '''
        Set the image that will be shown by the dialog window.

        Parameters
        ----------
        data: np.ndarray
            Image data as a numpy array.
        '''
        self.previewWidget.setImage(data)

    def exec_(self):
        '''
        Show modal dialog window with the image.
        '''
        if self._size is not None:
            self.resize(self._size)
        if self._position is not None:
            self.move(self._position)
        super().exec_()

        self._size = self.size()
        self._position = self.pos()

class SpecialImageItem(QWidget):
    refresh = QSignal()
    changed = QSignal(bool)

    def __init__(self, label='Special', **kwargs):
        '''
        Creates a widget for managing special image items such as an image of
        dark background or of a reference target.

        Parameters
        ----------
        label: str
            A short name of the item displayed in a label.
        kwargs: dict
            Arguments passed to the QWidget baseclass constructor.
        '''
        super().__init__(**kwargs)

        self._label = str(label)
        self._data = self._metadata = None
        self._viewDialog = ImageViewDialog(self)

        self.refreshButton = common.QSquarePushButton()
        self.refreshButton.setIcon(resources.loadIcon('refresh.png'))
        self.refreshButton.adjustSize()
        self.refreshButton.clicked.connect(self._refresh)
        self.refreshButton.setToolTip(
            QCoreApplication.translate('SpecialImageItem', 'Update')
        )
        self._pixmap_height = self.refreshButton.height()

        self.clearButton = common.QSquarePushButton()
        self.clearButton.setIcon(resources.loadIcon('recyclebin.png'))
        self.clearButton.clicked.connect(lambda x: self.clear())
        self.clearButton.setToolTip(
            QCoreApplication.translate('SpecialImageItem', 'Remove')
        )

        self._unavailableIcon = resources.loadPixmap(
            'unchecked.png').scaledToHeight(
                self._pixmap_height, Qt.SmoothTransformation)

        self._availableIcon = resources.loadPixmap(
            'checked.png').scaledToHeight(
                self._pixmap_height, Qt.SmoothTransformation)

        self.previewLabel = common.QClickableLabel()
        self.previewLabel.setPixmap(self._unavailableIcon)
        self.previewLabel.setFocusPolicy(Qt.StrongFocus)
        self.previewLabel.installEventFilter(self)
        self.previewLabel.doubleClicked.connect(self.showItemModal)

        self.descriptionLabel = QLabel(self._label)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.previewLabel)
        layout.addWidget(self.descriptionLabel)
        layout.addStretch()
        layout.addWidget(self.clearButton)
        layout.addWidget(self.refreshButton)
        self.setLayout(layout)

    def available(self) -> bool:
        '''
        Returns True if the special item holds valid data, else False.

        Returns
        --------
        available: bool
            True if special item holds valid data, else False.
        '''
        return self._data is not None

    def label(self) -> str:
        '''
        Returns
        -------
        label: str
            Short name of the special item.
        '''
        return self._label

    def data(self) -> np.ndarray:
        '''
        Returns
        -------
        data: np.ndarray
            Data associated with the special item.
        '''
        return self._data

    def setData(self, data: np.ndarray, metadata: dict = None):
        '''
        Set data associated with the special item.

        Parameters
        ----------
        data: np.ndarray
            Special item data.
        metadata: dict
            Metadata associated with the special item data.
        '''
        previous = self._data
        self._data = data
        self._metadata = metadata

        if previous is not self._data:
            self._updatePreviewPixmap()
            self._updateDescription()
            self.changed.emit(self.available())

    def metadata(self) -> dict:
        '''
        Returns
        -------
        metadata: np.ndarray
            Metadata associated with the special item.
        '''
        return self._metadata

    def clear(self):
        '''
        Drops/clears the data and metadata associated with the special item.
        '''
        if self.available():
            result = dialogs.QuestionMessage(
                self,
                QCoreApplication.translate('DataWidget', 'Replace'),
                QCoreApplication.translate(
                    'DataWidget', 'Delete existing item?')
            ).exec()
            if result != QMessageBox.Yes:
                return

        previous = self._data
        self._data = self._metadata = None
        if previous is not self._data:
            self._updateDescription()
            self._updatePreviewPixmap()
            self.changed.emit(self.available())

    def showItemModal(self):
        '''
        Show item data/image in a modal dialog window. Dialog is shown only
        if item holds valid data (the available method returns True)
        '''
        if self.available():
            data = self.data()
            self._viewDialog.setImage(data)
            self._viewDialog.setWindowTitle(
                '{} : {}/{}b : {}'.format(
                    self.label(), self._bitDepth(), data.itemsize*8,
                    self._shapeString(data.shape[::-1]))
                )
            self._viewDialog.exec_()

    def _createImagePreviewDialog(self):
        # create a modal preview dialog
        previewDialog = QDialog(self, Qt.Dialog)
        previewDialog.setWindowFlags(
            previewDialog.windowFlags() |
            Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        previewWidget = imview.Image()
        previewWidget.hideProfiles()
        layout = QVBoxLayout(previewDialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(previewWidget)
        previewDialog.previewWidget = previewWidget
        previewDialog.position = previewDialog.size = None

        return previewDialog

    def _shapeString(self, shape):
        shapestr = ''
        for item in shape:
            if shapestr:
                shapestr += ' x '
            shapestr += str(item)
        return shapestr

    def _bitDepth(self):
        bitdepth = self._data.itemsize*8
        if self._metadata is not None:
            candidate = self._metadata.get('bitdepth')
            if candidate is not None:
                bitdepth = candidate
        return bitdepth

    def _updateDescription(self):
        if self._data is not None:
            self.descriptionLabel.setText('{}\n{}/{}b\n{}'.format(
                self._label,
                self._bitDepth(), self._data.itemsize*8,
                self._shapeString(self._data.shape[::-1])))
        else:
            self.descriptionLabel.setText('{}'.format(self._label))

    def _updatePreviewPixmap(self):
        if not self.available():
            self.previewLabel.setPixmap(self._unavailableIcon)
        else:
            qimage = common.GsNp2QImage(self._data, self._bitDepth())[0]
            self._previewQImage = qimage
            self._previewPixmap = self._createPreviewPixmap(qimage)
            self.previewLabel.setScaledContents(False)
            self.previewLabel.setPixmap(self._previewPixmap)
            self._updateDescription()

    def _createPreviewPixmap(self, qimage):
        self._scaledPreviewImage = qimage.scaledToHeight(
            self._pixmap_height,
            Qt.SmoothTransformation
        )
        return QPixmap.fromImage(self._scaledPreviewImage)

    def _refresh(self):
        if self.available():
            result = dialogs.QuestionMessage(
                self,
                QCoreApplication.translate('DataWidget', 'Replace'),
                QCoreApplication.translate(
                    'DataWidget', 'Replace existing item?')
            ).exec()
            if result != QMessageBox.Yes:
                return

        self.refresh.emit()

    def clearForce(self):

        '''Clears data without asking.'''

        previous = self._data
        self._data = self._metadata = None
        if previous is not self._data:
            self._updateDescription()
            self._updatePreviewPixmap()
            self.changed.emit(self.available())


    def show_roi(self, switch: bool, pos=(0, 0), size=(100, 100), color='g'):
        self._viewDialog.previewWidget.toggleroi(switch, pos, size, color)

    def get_roi(self):
        return self._viewDialog.previewWidget._roi_rect

    def show_crosshair(self, key, switch: bool, pos=(0, 0), color='r'):
        self._viewDialog.previewWidget.togglecrosshair(key, switch, pos, color)

    def get_crosshair(self, key):
        return self._viewDialog.previewWidget._crosshairs.get(key)
    
    def get_crosshair_keys(self):
        return self._viewDialog.previewWidget._crosshairs.keys()

class SpecialTwoImageItem(QWidget):
    refresh = QSignal()
    changed = QSignal(bool)

    def __init__(self, label=('Special1', 'Special2'), **kwargs):
        '''
        Creates a widget for managing special image items such as an image of
        dark background or of a reference target.

        Parameters
        ----------
        label: (str, str)
            A short name of the item displayed in a label.
        kwargs: dict
            Arguments passed to the QWidget baseclass constructor.
        '''
        super().__init__(**kwargs)

        self._label = label
        self._data = self._metadata = None
        self._viewDialog = (ImageViewDialog(self), ImageViewDialog(self))

        self.refreshButton = common.QSquarePushButton()
        self.refreshButton.setIcon(resources.loadIcon('refresh.png'))
        self.refreshButton.adjustSize()
        self.refreshButton.clicked.connect(self._refresh)
        self.refreshButton.setToolTip(
            QCoreApplication.translate('SpecialImageItem', 'Update')
        )
        self._pixmap_height = self.refreshButton.height()

        self.clearButton = common.QSquarePushButton()
        self.clearButton.setIcon(resources.loadIcon('recyclebin.png'))
        self.clearButton.clicked.connect(lambda x: self.clear())
        self.clearButton.setToolTip(
            QCoreApplication.translate('SpecialImageItem', 'Remove')
        )

        self._unavailableIcon = resources.loadPixmap(
            'unchecked.png').scaledToHeight(
                self._pixmap_height, Qt.SmoothTransformation)

        self.previewLabel1 = common.QClickableLabel()
        self.previewLabel1.setPixmap(self._unavailableIcon)
        self.previewLabel1.setFocusPolicy(Qt.StrongFocus)
        self.previewLabel1.installEventFilter(self)
        self.previewLabel1.doubleClicked.connect(lambda: self.showItemModal(0))

        self.previewLabel2 = common.QClickableLabel()
        self.previewLabel2.setPixmap(self._unavailableIcon)
        self.previewLabel2.setFocusPolicy(Qt.StrongFocus)
        self.previewLabel2.installEventFilter(self)
        self.previewLabel2.doubleClicked.connect(lambda: self.showItemModal(1))

        self.descriptionLabel1 = QLabel(self._label[0])
        self.descriptionLabel2 = QLabel(self._label[1])

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        sublayout1 = QHBoxLayout()
        sublayout1.setContentsMargins(0, 0, 0, 0)
        sublayout1.addWidget(self.previewLabel1)
        sublayout1.addWidget(self.descriptionLabel1)
        sublayout2 = QHBoxLayout()
        sublayout2.setContentsMargins(0, 0, 0, 0)
        sublayout2.addWidget(self.previewLabel2)
        sublayout2.addWidget(self.descriptionLabel2)
        sublayout = QVBoxLayout()
        sublayout.setContentsMargins(0, 0, 0, 0)
        sublayout.addLayout(sublayout1)
        sublayout.addLayout(sublayout2)
        layout.addLayout(sublayout)
        layout.addStretch()
        layout.addWidget(self.clearButton)
        layout.addWidget(self.refreshButton)
        self.setLayout(layout)

    def available(self) -> bool:
        '''
        Returns True if the special item holds valid data, else False.

        Returns
        --------
        available: bool
            True if special item holds valid data, else False.
        '''
        return self._data is not None

    def label(self):
        '''
        Returns
        -------
        label: (str, str)
            Short name of the special item.
        '''
        return self._label

    def data(self):
        '''
        Returns
        -------
        data: (np.ndarray, np.ndarray)
            Data associated with the special item.
        '''
        return self._data

    def setData(self, data, metadata: dict = None):
        '''
        Set data associated with the special item.

        Parameters
        ----------
        data: (np.ndarray, np.ndarray)
            Special item data.
        metadata: dict
            Metadata associated with the special item data.
        '''
        previous = self._data
        self._data = data
        self._metadata = metadata

        if previous is not self._data:
            self._updatePreviewPixmap()
            self._updateDescription()
            self.changed.emit(self.available())

    def metadata(self) -> dict:
        '''
        Returns
        -------
        metadata: np.ndarray
            Metadata associated with the special item.
        '''
        return self._metadata

    def clear(self):
        '''
        Drops/clears the data and metadata associated with the special item.
        '''
        if self.available():
            result = dialogs.QuestionMessage(
                self,
                QCoreApplication.translate('DataWidget', 'Replace'),
                QCoreApplication.translate(
                    'DataWidget', 'Delete existing items?')
            ).exec()
            if result != QMessageBox.Yes:
                return

        previous = self._data
        self._data = self._metadata = None
        if previous is not self._data:
            self._updateDescription()
            self._updatePreviewPixmap()
            self.changed.emit(self.available())

    def showItemModal(self, index):
        '''
        Show item data/image in a modal dialog window. Dialog is shown only
        if item holds valid data (the available method returns True)
        '''
        if self.available():
            data = self.data()
            self._viewDialog[index].setImage(data[index])
            self._viewDialog[index].setWindowTitle(
                '{} : {}/{}b : {}'.format(
                    self.label()[index], self._bitDepth(index), data[index].itemsize*8,
                    self._shapeString(data[index].shape[::-1]))
                )
            self._viewDialog[index].exec_()

    def _shapeString(self, shape):
        shapestr = ''
        for item in shape:
            if shapestr:
                shapestr += ' x '
            shapestr += str(item)
        return shapestr

    def _bitDepth(self, index):
        bitdepth = self._data[index].itemsize*8
        return bitdepth

    def _updateDescription(self):
        if self._data is not None:
            self.descriptionLabel1.setText('{}\n{}/{}b\n{}'.format(
                self._label[0],
                self._bitDepth(0), self._data[0].itemsize*8,
                self._shapeString(self._data[0].shape[::-1])))
            self.descriptionLabel2.setText('{}\n{}/{}b\n{}'.format(
                self._label[1],
                self._bitDepth(1), self._data[1].itemsize*8,
                self._shapeString(self._data[1].shape[::-1])))
        else:
            self.descriptionLabel1.setText('{}'.format(self._label[0]))
            self.descriptionLabel2.setText('{}'.format(self._label[1]))

    def _updatePreviewPixmap(self):
        if not self.available():
            self.previewLabel1.setPixmap(self._unavailableIcon)
            self.previewLabel2.setPixmap(self._unavailableIcon)
        else:
            qimage1 = common.GsNp2QImage(self._data[0], self._bitDepth(0))[0]
            qimage2 = common.GsNp2QImage(self._data[1], self._bitDepth(1))[0]
            self._previewQImage1 = qimage1
            self._previewQImage2 = qimage2
            self._previewPixmap1 = self._createPreviewPixmap(qimage1)
            self._previewPixmap2 = self._createPreviewPixmap(qimage2)
            self.previewLabel1.setScaledContents(False)
            self.previewLabel1.setPixmap(self._previewPixmap1)
            self.previewLabel2.setScaledContents(False)
            self.previewLabel2.setPixmap(self._previewPixmap1)
            self._updateDescription()

    def _createPreviewPixmap(self, qimage):
        self._scaledPreviewImage = qimage.scaledToHeight(
            self._pixmap_height,
            Qt.SmoothTransformation
        )
        return QPixmap.fromImage(self._scaledPreviewImage)

    def _refresh(self):
        if self.available():
            result = dialogs.QuestionMessage(
                self,
                QCoreApplication.translate('DataWidget', 'Replace'),
                QCoreApplication.translate(
                    'DataWidget', 'Replace existing items?')
            ).exec()
            if result != QMessageBox.Yes:
                return

        self.refresh.emit()

class FFTImageSupervisor(QWidget):

    checkboxchanged = QSignal()

    def __init__(self, label='FFT', **kwargs):
        '''
        Creates a widget for managing FFT

        Parameters
        ----------
        label: str
            A short name of the item displayed in a label.
        kwargs: dict
            Arguments passed to the QWidget baseclass constructor.
        '''
        super().__init__(**kwargs)

        self._label = str(label)

        self.states = {
            'dark': False,
            'reference': False,
            'object': False,
            'divisor': False
        }

        self.darkcheckbox = QCheckBox('Dark')
        self.darkcheckbox.stateChanged.connect(self.check_states)
        self.referencecheckbox = QCheckBox('Reference')
        self.referencecheckbox.stateChanged.connect(self.check_states)
        self.objectcheckbox = QCheckBox('Object')
        self.objectcheckbox.stateChanged.connect(self.check_states)
        self.divisorcheckbox = QCheckBox('Divisor')
        self.divisorcheckbox.stateChanged.connect(self.check_states)

        layout1 = QGridLayout()
        layout1.addWidget(self.darkcheckbox, 0, 0)
        layout1.addWidget(self.referencecheckbox, 0, 1)
        layout1.addWidget(self.objectcheckbox, 1, 0)
        layout1.addWidget(self.divisorcheckbox, 1, 1)

        layout2 = QVBoxLayout()
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.addLayout(layout1)

        self.setLayout(layout2)

    def check_states(self):

        self.states['dark'] = self.darkcheckbox.checkState()
        self.states['reference'] = self.referencecheckbox.checkState()
        self.states['object'] = self.objectcheckbox.checkState()
        self.states['divisor'] = self.divisorcheckbox.checkState()
        self.checkboxchanged.emit()
        # TODO: force the viewmodel to start showing the correct FFT

    def label(self) -> str:
        '''
        Returns
        -------
        label: str
            Short name of the special item.
        '''
        return self._label

class MaxFinder(QWidget):
    rectroiboxchanged = QSignal(bool)
    maxboxchanged = QSignal(bool)

    def __init__(self, label='FFT', **kwargs):
        '''
        Creates a widget for managing FFT

        Parameters
        ----------
        label: str
            A short name of the item displayed in a label.
        kwargs: dict
            Arguments passed to the QWidget baseclass constructor.
        '''
        super().__init__(**kwargs)

        self._label = str(label)
        self._data = self._metadata = None

        self._rectroi_state = False
        self._max_state = False

        self.rectroi = QCheckBox('Rect roi')
        self.rectroi.stateChanged.connect(self.rectroi_state)

        self.max = QCheckBox('Find max')
        self.max.clicked.connect(self.max_find)

        layout1 = QGridLayout()
        layout1.addWidget(self.darkcheckbox, 0, 0)
        layout1.addWidget(self.referencecheckbox, 0, 1)
        layout1.addWidget(self.objectcheckbox, 1, 0)
        layout1.addWidget(self.divisorcheckbox, 1, 1)

        layout2 = QVBoxLayout()
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.addLayout(layout1)

        self.setLayout(layout2)

    def rectroi_state(self):
        self._rectroi_state = self.rectroi.checkState()
        self.rectroiboxchanged.emit(self._rectroi_state)

        if not self._rectroi_state:
            self.max.setCheckState(False)

    def max_find(self):
        if self._rectroi_state:
            self._max_state = self.max.checkState()
            self.maxboxchanged.emit(self._max_state)
        else:
            self.max.setCheckState(False)



    def check_states(self):

        self.states['dark'] = self.darkcheckbox.checkState()
        self.states['reference'] = self.referencecheckbox.checkState()
        self.states['object'] = self.objectcheckbox.checkState()
        self.states['divisor'] = self.divisorcheckbox.checkState()
        self.checkboxchanged.emit()
        # TODO: force the viewmodel to start showing the correct FFT

    def label(self) -> str:
        '''
        Returns
        -------
        label: str
            Short name of the special item.
        '''
        return self._label

class ViewModel:
    def __init__(self, label: str, tooltip: str, requires=None):
        '''
        Base class of view models. Returns the latest sample by
        default.
        Parameters
        ----------
        label: str
            Short name of the view model used in the UI.
        tooltip: str
            Tooltip used with the view model.
        required: str, tuple, list
            A list of required special item identifiers.
        '''
        if requires is None:
            requires = ()
        self._label = label
        self._tooltip = tooltip
        self._requires = tuple(requires)

    def __call__(self, datamodule) -> np.ndarray:
        '''
        Returns
        -------
        processed: np.ndarray
            Processed sample.
        '''
        return datamodule.data()

    def label(self) -> str:
        '''
        Returns
        -------
        name: str
            View model label.
        '''
        return self._label

    def tooltip(self) -> str:
        '''
        Returns
        -------
        tooltip: str
            View model tooltip.
        '''
        return self._tooltip

    def requires(self) -> tuple:
        '''
        Returns
        -------
        required: tuple
            A tuple of required special item identifiers.
        '''
        return self._requires

    def canApply(self, datamodule) -> bool:
        '''
        Returns True if the view model can be applied to the data module
        (checks that all special items are managed by the data module
        and available)
        Returns
        -------
        canapply: bool
            True if the view model can be applied to the data module,
            else False.
        '''
        for item_identifier in self._requires:
            item = datamodule.specialItem(item_identifier)
            if item is None or not item.available():
                return False
        return True

class RawViewModel(ViewModel):
    def __init__(self):
        super().__init__(
            label=QCoreApplication.translate('RawViewModel', 'Raw'),
            tooltip=QCoreApplication.translate(
                'RawViewModel', 'Unprocessed raw sample')
        )

class SubtractDarkViewModel(ViewModel):
    def __init__(self):
        '''
        A view model that subtracts dark background from the sample.
        Requires a data module with available "dark" special item.
        '''
        super().__init__(
            label=QCoreApplication.translate(
                'SubtractDark', 'Dark subtracted'),
            tooltip=QCoreApplication.translate(
                'SubtractDark', 'Subtract dark background from the sample'),
            requires=('dark',)
        )

    def __call__(self, datamodule) -> np.ndarray:
        dark = datamodule.specialItem('dark').data()
        sample = np.array(datamodule.data())

        mask = dark > sample
        sample -= dark
        sample[mask] = 0

        return sample

class ReflectanceViewModel(ViewModel):
    def __init__(self):
        '''
        A view model that computes reflectance from the sample.
        Requires a data module with available "dark" and "reference" special
        items.
        '''
        super().__init__(
            label=QCoreApplication.translate('Reflectance', 'Reflectance'),
            tooltip=QCoreApplication.translate(
                'Reflectance', 'Compute reflectance'),
            requires=('dark', 'reference')
        )

    def __call__(self, datamodule) -> np.ndarray:
        dark = datamodule.specialItem('dark').data().astype(np.float64)
        sample = np.array(datamodule.data(), dtype=np.float64)
        reference = np.array(
            datamodule.specialItem('reference').data(), dtype=np.float64)

        sample -= dark
        reference -= dark
        reference[reference == 0.0] = np.inf

        sample /= reference

        return sample

# DHM preprocessing and viewmodels
class SubtractDivideSignalsViewModel(ViewModel):
    def __init__(self):
        '''
        A view model that subtracts and divides available signals
        from the sample. Requires a data module with available "dark",
        "reference", "object" and "divisor" items.
        '''
        super().__init__(
            label=QCoreApplication.translate('SubtractDivide', 'SubtractDivide'),
            tooltip=QCoreApplication.translate(
                'SubtractDivide', 'Compute subtraction or division'),
            requires=('dark', 'reference', 'object', 'divisor')
        )

    def __call__(self, datamodule) -> np.ndarray:
        sample = np.array(datamodule.data(), dtype=np.float64)
        for item_identifier in self._requires:
            item = datamodule.specialItem(item_identifier)
            if item.available() and item_identifier != 'divisor':
                sample -= item.data()
            elif item.available():
                sample /= item.data()

        return sample
    
    def canApply(self, datamodule) -> bool:
        return True

class DataHubModule(fpa.HubModule):
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self._lastSample = None
        self._saveFileName = None
        self._fileCounter = 0
        self._acquisition = None
        self._device = device
        self._specialItems = {}

        # Module tooltip.
        self._description = QCoreApplication.translate(
            'DataHubModule', 'Data processing and acquisition module'
        )
        # Module tooltip.
        self._tooltip = QCoreApplication.translate(
            'DataHubModule', 'Data processing and acquisition module'
        )
        # A short name of the Module.
        self._label = QCoreApplication.translate(
            'DataHubModule', 'Process'
        )
        # Module icon.
        self._icon = resources.loadIcon('preview.png')

        # layout that will hold special image items
        self._specialItemsLayout = QVBoxLayout()
        self._specialItemsLayout.setContentsMargins(0, 0, 0, 0)

        # create and add standard special items
        darkItem = SpecialImageItem(
            QCoreApplication.translate('DataHubModule', 'Dark')
        )
        self.addSpecialItem('dark', darkItem)

        referenceItem = SpecialImageItem(
            QCoreApplication.translate('DataHubModule', 'Reference')
        )
        self.addSpecialItem('reference', referenceItem)

        # view mode
        self.viewModelComboBox = QComboBox()
        self.viewModelComboBox.setView(QListView())  # required by stylesheet
        self.viewModelComboBox.setToolTip(
            QCoreApplication.translate(
                'DataHubModule', 'Select a view/processing mode.'
            )
        )
        self.viewModeLabel = QLabel(
            QCoreApplication.translate('DataHubModule', 'View model'),
        )
        self.viewModeLabel.setBuddy(self.viewModelComboBox)
        self.viewModelComboBox.currentIndexChanged.connect(
            self.setViewModelIndex)
        viewModeLayout = QHBoxLayout()
        viewModeLayout.addWidget(self.viewModelComboBox)
        viewModeLayout.addWidget(self.viewModeLabel)

        # create and add standard view models
        self.addViewModel(RawViewModel())
        self.addViewModel(SubtractDarkViewModel())
        self.addViewModel(ReflectanceViewModel())
        self.viewModelComboBox.setCurrentIndex(0)

        self.fileSelectButton = QPushButton(
            QCoreApplication.translate(
                'DataHubModule', 'Select output file and capture mode'
            )
        )
        self.fileSelectButton.setToolTip(
            QCoreApplication.translate(
                'DataHubModule',
                'Select output file using a standard file dialog.'
            ),
        )
        self.fileSelectButton.clicked.connect(lambda x: self.setSaveFileName())

        self.fileLineEdit = QLineEdit()
        self.fileLineEdit.setToolTip(
            QCoreApplication.translate(
                'DataHubModule',
                'The selected output file with a managed suffix.'
            ),
        )

        self.acquireButton = QPushButton(
            QCoreApplication.translate(
                'DataHubModule', 'Capture'
            )
        )
        self.acquireButton.setIcon(resources.loadIcon('acquire.png'))
        self.acquireButton.setToolTip(
            QCoreApplication.translate(
                'DataHubModule',
                'Acquire a set of spectra that include dark background, '\
                'reference and sample (NPZ, MAT or HDF5 file) '
                'or continuously record the stream (ACQ file).'),
        )
        self.acquireButton.clicked.connect(lambda x: self.capture())

        intensity_span = (0, 255)
        if device is not None:
            intensity_span = (0, 2**device.get('bitdepth') - 1)
        self._viewWidget = imview.Image(levels=intensity_span, parent=self)
        self._viewWidget.setRange(intensity_span)
        self._viewWidget.setLevels(intensity_span)
        self._viewWidget.hideProfiles()

        layout = QVBoxLayout()
        layout.addLayout(self._specialItemsLayout)
        layout.addWidget(common.horizontalLine())
        layout.addLayout(viewModeLayout)
        layout.addWidget(common.horizontalLine())
        layout.addStretch()
        layout.addWidget(self.fileSelectButton)
        layout.addWidget(self.fileLineEdit)
        layout.addWidget(self.acquireButton)

        self.setLayout(layout)

    def available(self) -> bool:
        '''
        Returns
        -------
        available: bool
            True if sample data is available, else False.
        '''
        return self._lastSample is not None

    def data(self) -> np.ndarray:
        '''
        Returns
        -------
        data: np.ndarray
            Last sample data.
        '''
        return self._lastSample['data']

    def metadata(self) -> dict:
        '''
        Returns
        -------
        metadata: dict
            Last sample metadatadata.
        '''
        return self._lastSample['data']

    def specialItem(self, identifier: str) -> SpecialImageItem:
        '''
        Returns special item for the given identifier.

        Parameters
        ----------
        identifer: str
            Special item identifier.

        Returns
        -------
        item: SpecialImageItem
            Special item instance.
        '''
        return self._specialItems[identifier]

    def addSpecialItem(self, identifier: str, item: SpecialImageItem,
                       index: int = -1):
        '''
        Add special item to the widget.

        Parameters
        ----------
        identifier: str
            Special item identifier.
        item SpecialImageItem
            Special item to add.
        index: int
            Insert position of the special item in the layout.
        '''
        if identifier in self._specialItems:
            old_item = self._specialItems.pop(identifier)
            old_item.refresh.disconnect(self.updateSpecialItem)
            self._specialItemsLayout.remove(old_item)

        self._specialItems[identifier] = item
        if index == -1:
            self._specialItemsLayout.addWidget(item)
        else:
            self._specialItemsLayout.insertWidget(index, item)

        item.refresh.connect(lambda: self.updateSpecialItem(item))

    def removeSpecialItem(self, item: str or SpecialImageItem):
        '''
        Remeve special item from the widget.

        Parameters
        ----------
        item: SpecialImageItem
            Special item to remove.
        '''
        if isinstance(item, SpecialImageItem):
            self._specialIemWidget.remove(item)
            self._specialItems.pop(self.specialItemIdentifier(item))
        else:
            self._specialItemsLayout.remove(self._specialItems[item])
            self._specialItems.pop(item)

        item.refresh.disconnect(self.updateSpecialItem)

    def specialItemIdentifier(self, item: SpecialImageItem) -> str:
        '''
        Returns special item identifier or None if the special item is not
        managed by this widget.

        Parameters
        ----------
        item: SpecialImageItem
            Special item widget.

        Returns
        -------
        identifier: str
            Special item identifier.
        '''
        identifier = None
        for key, special_item in self._specialItems.items():
            if special_item == item:
                identifier = key
        return identifier

    def updateSpecialItem(self, item):
        '''
        Update data of the special item with the current sample.

        Parameters
        ----------
        identifier: str or SpecialImageItem
            Special item or special item identifier
        '''
        if not isinstance(item, SpecialImageItem):
            item = self._specialItems[item]
        item.setData(self._lastSample['data'], self._lastSample['metadata'])

    def addViewModel(self, model: ViewModel, index: int = -1):
        '''
        Adds a view model to the data module.

        Parameters
        ----------
        model: ViewModel
            Vie model instance.
        index: int
            Insert position. Use -1 (default) to append the new view model at
            the end of the list.
        '''
        if index == -1:
            self.viewModelComboBox.addItem(model.label(), model)
        else:
            self.viewModelComboBox.insertItem(
                index, model.label(), model)

    def removeViewModel(self, model: ViewModel or int):
        '''
        Remove view model from the data module.
        Parameters
        ----------
        processor: int or ViewModel
            Removes the given view model.
        '''
        remove_index = None
        if isinstance(model, ViewModel):
            for index in range(self.viewModelComboBox.count()):
                if model == self.viewModelComboBox.itemData(index):
                    remove_index = index
                    break
        else:
            remove_index = model
        self.viewModelComboBox.removeItem(remove_index)

    def connectHub(self, hub):
        super().connectHub(hub)
        device = hub.device('camera')
        if device is None:
            dialogs.ErrorMessage(
                self,
                QCoreApplication.translate(
                    'DataHubModule', 'Connect to hub error'
                ),
                QCoreApplication.translate(
                    'DataHubModule',
                    'This module requires a camera capable hub!'
                )
            ).exec()
        self._device = device

        intensity_span = (0, 2**self._device.get('bitdepth') - 1)
        self._viewWidget.setRange(intensity_span)
        self._viewWidget.setLevels(intensity_span)

        hub.newPreviewItem.connect(self.updateData)

    def disconnectHub(self):
        if self.hub() is not None:
            self._device = None
            self.hub().newPreviewItem.disconnect(self.updateData)
        super().disconnectHub()

    def icon(self):
        return self._icon

    def label(self):
        return self._label

    def tooltip(self):
        return self._tooltip

    def description(self):
        return self._description

    def view(self):
        return self._viewWidget

    def updateData(self, item):
        data, index, metadata = item
        self._lastSample = {'data':data, 'metadata':metadata, 'index':index}

        self.process()

    def setViewModelIndex(self, index: int):
        '''
        Set the current view model.

        Parameters
        ----------
        index: int
            View model index in the ComboBox list.
        '''
        index = clip(index, 0, self.viewModelComboBox.count() - 1)
        model = self.viewModelComboBox.itemData(index)
        if not model.canApply(self):
            self.viewModelComboBox.setCurrentIndex(0)
            ui_list = []
            for item_identifier in model.requires():
                ui_list.append(self.specialItem(item_identifier).label())

            dialogs.ErrorMessage(
                self,
                QCoreApplication.translate(
                    'DataHubModule', 'View model error'
                ),
                QCoreApplication.translate(
                    'DataHubModule',
                    'The selected view model requires the following items: '\
                    '{:s}!'.format(', '.join(ui_list))
                )
            ).exec()

    def viewModelIndex(self) -> int:
        '''
        Returns
        -------
        index: int
            Currently selected view model index.
        '''
        return self.viewModelComboBox.currentIndex()

    def viewModel(self) -> ViewModel:
        '''
        Returns
        -------
        model: ViewModel
            Currently selected view model.
        '''
        return self.viewModelComboBox.currentData()

    def process(self):
        model = self.viewModel()
        if not model.canApply(self):
            self.setViewModelIndex(0)
        else:
            processed_sample = model(self)
            self._updateView(processed_sample)

    def setSaveFileName(self, filename=None):
        if filename is None:
            filename = QFileDialog.getSaveFileName(
                self,
                QCoreApplication.translate('DataWidget', 'Save to file'),
                '',
                QCoreApplication.translate(
                    'DataHubModule',
                    'Snapshot capture to compressed numpy (*.npz);;'\
                    'Snapshot capture to compressed mat (*.mat);;'\
                    'Snapshot capture to HDF5 (*.h5);;'\
                    'Continuous recording to picked (*.acq)'),
                None, QFileDialog.DontConfirmOverwrite
            )
            if isinstance(filename, tuple):
                filename = filename[0]

        if filename:
            try:
                base, ext = os.path.splitext(filename)
                new_base, end = base.rsplit('_')
                int(end)
                filename = '{:s}{:s}'.format(new_base, ext)
            except ValueError:
                pass

            self._saveFileName = filename
            self._fileCounter = 0

            self.fileLineEdit.setText(self._getNextFileName())

    def capture(self, filename=None):
        if self._acquisition:
            self.captureContinuous(filename)
            # display next available file
            self.fileLineEdit.setText(self._getNextFileName())
        else:
            if filename is None:
                filename = self._getNextFileName()
                if not filename:
                    self.setSaveFileName()
                    filename = self._getNextFileName()

            if filename:
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.acq':
                    self.captureContinuous(filename)
                    # next available file will be shown after stopping
                    # started acquisition
                elif ext in ('.npz', '.mat', '.h5'):
                    self.captureSingle(filename)
                    # display next available file
                    self.fileLineEdit.setText(self._getNextFileName())

    def _packData(self, items=()):
        data = {}
        for item in items:
            data[item] = []

        if 'dark' in data and self.specialItem('dark').available():
            dark = self.specialItem('dark')
            data['dark'] = dark.data()
            metadata = dark.metadata()
            if metadata is not None:
                data['dark_metadata'] = metadata
        if 'reference' in data and self.specialItem('reference').available():
            reference = self.specialItem('reference')
            data['reference'] = reference.data()
            metadata = reference.metadata()
            if metadata is not None:
                data['reference_metadata'] = metadata
        if 'sample' in data and self.available() is not None:
            data['sample'] = self.data()
            metadata = self.metadata()
            if metadata is not None:
                data['sample_metadata'] = metadata

        return data

    def captureSingle(self, filename):
        if filename:
            data = self._packData(('dark', 'reference', 'sample'))

            ext = os.path.splitext(filename)[-1].lower()
            options = {'mat':{'do_compression':True},
                       'pkl':{'protocol':PICKLE_PROTOCOL}}
            try:
                export.exportData(filename, ext, options, **data)
            except:
                dialogs.ErrorMessage(
                    self,
                    QCoreApplication.translate(
                        'DataHubModule', 'Save error'
                    ),
                    QCoreApplication.translate(
                        'DataHubModule',
                        'Failed to save captured data to the specified file!'
                    ),
                    traceback.format_exc()
                ).exec()

    def captureContinuous(self, filename):
        if self._acquisition is not None:
            self._device.stop()
            self._acquisition.set('close')
            self._acquisition = None
            self.acquireButton.setText(
                QCoreApplication.translate('DataHubModule', 'Capture'),
            )
            self.acquireButton.setIcon(resources.loadIcon('acquire.png'))
            self._device.acquire()
        else:
            if filename:
                header = self._packData(('dark', 'reference', 'sample'))

                try:
                    exporter = acquisition.Exporter(filename, header=header)
                except:
                    dialogs.ErrorMessage(
                        self,
                        QCoreApplication.translate(
                            'DataHubModule', 'Save error'
                        ),
                        QCoreApplication.translate(
                            'DataHubModule',
                            'Failed to start capturing to the specified file!'
                        ),
                        traceback.format_exc()
                    ).exec()

                self._acquisition = acquisition.Acquisition(
                    'inf', ofile=exporter)

                self._device.acquire(self._acquisition)

                self.acquireButton.setText(
                    QCoreApplication.translate('DataHubModule', 'Stop'),
                )
                self.acquireButton.setIcon(resources.loadIcon('cancel.png'))

    def _getNextFileName(self):
        if self._saveFileName is not None:
            base, ext = os.path.splitext(self._saveFileName)

            while True:
                filename = '{}_{}{}'.format(base, self._fileCounter, ext)
                if not os.path.exists(filename):
                    break
                self._fileCounter += 1

            return filename

    def _updateView(self, data):
        self._viewWidget.setImage(data, info=str(self._lastSample['index']))

    def _canApplyProcessing(self, previewModel):
        if previewModel == 0:
            return True
        elif previewModel == 1 and self.specialItem('dark').available():
            return True
        elif previewModel == 2 and self.specialItem('reference').available() \
                and self.specialItem('reference').available():
            return True
        return False

class CameraModule(fpa.FpaHubModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._view = self._createView(self.device())

        # prepare some default UI components for the hub
        self._icon = resources.loadIcon('camera.png')
        self._label = QCoreApplication.translate(
            'CameraModule',
            'Camera',
        )
        self._tooltip = QCoreApplication.translate(
            'HubModule',
            'Camera control module',
        )
        self._description = QCoreApplication.translate(
            'HubModule',
            'Camera control module',
        )

        self.newPreviewItem.connect(self.updateView)

    def _createView(self, camera):
        intensity_span = (0, 2**camera.get('bitdepth') - 1)
        view = imview.Image(levels=intensity_span)
        view.setRange(intensity_span)
        view.setLevels(intensity_span)
        view.hideProfiles()

        return view

    def updateView(self, item):
        # unpack preview item
        frame, frame_index, metadata = item
        self._view.setImage(frame, info='Frame {:d}'.format(frame_index))

    def icon(self):
        return self._icon

    def view(self):
        return self._view

    def label(self):
        return self._label

    def description(self):
        return self._description

    def tooltip(self):
        return self._tooltip

class CameraHub(fpa.Hub):
    exposureTimeChanged = QSignal(object)
    averagingChanged = QSignal(object)
    newPreviewItem = QSignal(object)

    def __init__(self, camera, cfg=None, **kwargs):
        '''
        Creates a hub with a camera control module.

        Parameters
        ----------
        camera: device.Device
            Camera device instance.
        cfg: dict
            Hub configuration. See DEFAULT_CONFIGURATION for more details.
        kwargs: dict
            Parameters passed to the base class constructor.
        '''
        super().__init__(**kwargs)

        self.addDevice('camera', camera)

        if cfg is None:
            cfg = DEFAULT_CONFIGURATION
        else:
            cfg = misc.mergeTwoDictsDeep(DEFAULT_CONFIGURATION, cfg)
        self.configuration = cfg

        self.cameraModule = CameraModule(camera, cfg=cfg)
        self.connectModule(self.cameraModule)

        self.cameraModule.exposureTimeChanged.connect(self.exposureTimeChanged)
        self.cameraModule.averagingChanged.connect(self.averagingChanged)
        self.cameraModule.newPreviewItem.connect(self.newPreviewItem.emit)

    def sizeHint(self):
        '''
        Hint for the initial size of the hub window.
        '''
        return QSize(1024, 600)

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

    camera = None
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
        camera = pylon.Pylon(pylon.Configuration(serial))
        if camera is not None:
            descriptor = camera.descriptor()
            title = 'Basler {:s} "{:s}" viewer'.format(
                descriptor.prettyName, descriptor.serial)

    if camera is not None:
        # print some basic information
        # print(title, camera.descriptor())

        # creating camera hub
        hub = CameraHub(camera, cfg=uicfg)
        hub.setWindowTitle(title)
        # adding a basic image processing and acquisition module
        sdm = DataHubModule()
        hub.connectModule(sdm)
        hub.show()

        app.exec_()
