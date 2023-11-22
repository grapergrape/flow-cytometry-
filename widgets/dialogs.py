# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 02:11:38 2017

@author: miran
"""

import sys

from PyQt5.QtWidgets import QWidget, QDialog, QHBoxLayout, \
                            QVBoxLayout, QLabel, QProgressBar, \
                            QPushButton, QPlainTextEdit, QSizePolicy,\
                            QListWidget, QListWidgetItem, QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication, QEvent
from PyQt5.QtCore import pyqtSignal as QSignal


from widgets import resources

def ErrorMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Critical, title, label,
                      QMessageBox.Ok, parent)
    dlg.button(QMessageBox.Ok).setIcon(resources.loadIcon('ok.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def ErrorQuestion(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Critical, title, label,
                      QMessageBox.Yes | QMessageBox.No, parent)
    dlg.button(QMessageBox.Yes).setIcon(resources.loadIcon('ok.png'))
    dlg.button(QMessageBox.No).setIcon(resources.loadIcon('cancel.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def WarningMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Warning, title, label,
                      QMessageBox.Ok, parent)
    dlg.button(QMessageBox.Ok).setIcon(resources.loadIcon('ok.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def InformationMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Information, title, label,
                      QMessageBox.Ok, parent)
    dlg.button(QMessageBox.Ok).setIcon(resources.loadIcon('ok.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def QuestionMessage(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Question, title, label,
                      QMessageBox.Yes | QMessageBox.No, parent)
    dlg.button(QMessageBox.Yes).setIcon(resources.loadIcon('ok.png'))
    dlg.button(QMessageBox.No).setIcon(resources.loadIcon('cancel.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

def WarningQuestion(parent, title, label, details=None):
    dlg = QMessageBox(QMessageBox.Warning, title, label,
                      QMessageBox.Yes | QMessageBox.No, parent)
    dlg.button(QMessageBox.Yes).setIcon(resources.loadIcon('ok.png'))
    dlg.button(QMessageBox.No).setIcon(resources.loadIcon('cancel.png'))
    if details is not None:
        dlg.setDetailedText(details)
    dlg.setWindowModality(Qt.WindowModal)
    return dlg

class QTaskProgressDialog(QDialog):
    SUCCESS = 1
    ERROR = -1
    WARNING = 0
    canceled = QSignal()

    def __init__(self, parent=None, title=None, label=None, pixmap=None):
        QDialog.__init__(self, parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._autoclose = True

        self.setWindowTitle(title)

        buttonLayout = QHBoxLayout()
        self._cancelButton = QPushButton(
            QCoreApplication.translate('QTaskProgressDialog', 'Cancel'))
        self._cancelButton.clicked.connect(self._buttonClicked)
        self._detailsButton = QPushButton(
            QCoreApplication.translate('QTaskProgressDialog', 'Details'))
        self._detailsButton.clicked.connect(self._showDetails)
        buttonLayout.addWidget(self._detailsButton)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self._cancelButton)

        labelLayout = QHBoxLayout()
        self._pixmapLabel = QLabel()
        self._pixmapLabel.setSizePolicy(QSizePolicy.Minimum,
                                        QSizePolicy.Minimum)
        if pixmap is not None:
            self._pixmapLabel.setPixmap(pixmap)
        else:
            self._pixmapLabel.hide()
        labelLayout.addWidget(self._pixmapLabel)
        self._label = QLabel(label)
        self._label.setSizePolicy(QSizePolicy.MinimumExpanding,
                                  QSizePolicy.Minimum)
        labelLayout.addWidget(self._label)

        mainVLayout = QVBoxLayout()
        self._progressBar = QProgressBar()
        self._progressBar.setRange(0, 100)
        mainVLayout.addWidget(self._progressBar)
        mainVLayout.addLayout(labelLayout)
        self._detailsList = QListWidget()
        self._detailsList.setSizePolicy(QSizePolicy.MinimumExpanding,
                                        QSizePolicy.MinimumExpanding)
        mainVLayout.addLayout(buttonLayout)
        self._detailsList.hide()
        mainVLayout.addWidget(self._detailsList, 1)
        mainVLayout.addStretch(0)
        self.setLayout(mainVLayout)

        self._success_icon = resources.loadIcon('success.png')
        self._warning_icon = resources.loadIcon('warning.png')
        self._error_icon = resources.loadIcon('error.png')

    def closeEvent(self, event):
        if self._progressBar.value() < self._progressBar.maximum():
            self.cancel()

    def changeEvent(self, event):
        if event.type() == QEvent.LanguageChange:
            self._cancelButton.setText(
                QCoreApplication.translate('QTaskProgressDialog', 'Cancel')
            )
            self._detailsButton.setText(
                QCoreApplication.translate('QTaskProgressDialog', 'Details')
            )
        else:
            QWidget.changeEvent(self, event)

    def setHelp(self, text):
        self.setWhatsThis(text)

    def _showDetails(self):
        self._detailsList.setVisible(not self._detailsList.isVisible())
        self.adjustSize()

    def _buttonClicked(self):
        if self.value() >= self.maximum():
            self.accept()
        else:
            self.cancel()

    def setShowDetails(self, state):
        self._detailsList.setVisible(state)
        self.adjustSize()

    def showDetails(self):
        self._detailsList.show()
        self.adjustSize()

    def hideDetails(self):
        self._detailsList.hide()
        self.adjustSize()

    def cancelButton(self):
        return self._cancelButton

    def detailsButton(self):
        return self._detailsButton

    def setIconPixmap(self, pixmap):
        self._pixmapLabel.setPixmap(pixmap)

    def accept(self):
        self.setValue(100)
        QDialog.accept(self)

    def cancel(self):
        self.canceled.emit()
        QDialog.reject(self)

    def reset(self):
        self.setResult(0)
        self._label.clear()
        self._detailsList.clear()
        self._detailsList.hide()
        self.adjustSize()
        self._progressBar.reset()
        self._cancelButton.setText(
            QCoreApplication.translate('QTaskProgressDialog', 'Cancel'))
        self.setVisible(not self._autoclose)

    def autoClose(self):
        return self._autoclose

    def setAutoclose(self, state):
        self._autoclose = bool(state)

    def value(self):
        return self._progressBar.value()

    def setMinimum(self, value):
        return self._progressBar.setMinimum(value)

    def minimum(self):
        return self._progressBar.minimum()

    def setMaximum(self, value):
        return self._progressBar.setMaximum(value)

    def maximum(self):
        return self._progressBar.maximum()

    def setRange(self, minimum, maximum):
        return self._progressBar.setRange(minimum, maximum)

    def setValue(self, value):
        self._progressBar.setValue(value)
        if value >= self._progressBar.maximum():
            if self._autoclose:
                QDialog.accept(self)
            self._cancelButton.setText(
                QCoreApplication.translate('QTaskProgressDialog', 'Close'))

    def labelText(self):
        return self._label.text()

    def setLabelText(self, text):
        return self._label.setText(text)

    def appendItem(self, text, success=SUCCESS):
        if success == QTaskProgressDialog.SUCCESS:
            item = QListWidgetItem(self._success_icon, text)
        elif success == QTaskProgressDialog.WARNING:
            item = QListWidgetItem(self._warning_icon, text)
        elif success == QTaskProgressDialog.ERROR:
            item = QListWidgetItem(self._error_icon, text)
        else:
            item = QListWidgetItem(text)

        self._detailsList.addItem(item)
        self._detailsList.scrollToItem(
            self._detailsList.item(self._detailsList.count() - 1))

    def exec(self):
        QDialog.exec(self)
        return self._progressBar.value()

class QProgressDialogEx(QDialog):
    canceled = QSignal()

    def __init__(self, parent=None, title=None, label=None, pixmap=None):
        QDialog.__init__(self, parent)

        self._autoclose = True

        self.setWindowTitle(title)

        buttonLayout = QHBoxLayout()
        self._cancelButton = QPushButton(
            QCoreApplication.translate('QProgressDialogEx', 'Cancel'))
        self._cancelButton.clicked.connect(self._buttonClicked)
        self._detailsButton = QPushButton(
            QCoreApplication.translate('QProgressDialogEx', 'Details'))
        self._detailsButton.clicked.connect(self._showDetails)
        buttonLayout.addWidget(self._detailsButton)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self._cancelButton)

        labelLayout = QHBoxLayout()
        self._pixmapLabel = QLabel()
        self._pixmapLabel.setSizePolicy(QSizePolicy.Minimum,
                                        QSizePolicy.Minimum)
        if pixmap is not None:
            self._pixmapLabel.setPixmap(pixmap)
        else:
            self._pixmapLabel.hide()
        self._label = QLabel(label)
        self._label.setSizePolicy(QSizePolicy.MinimumExpanding,
                                  QSizePolicy.Minimum)
        labelLayout.addWidget(self._pixmapLabel)
        labelLayout.addWidget(self._label)

        mainVLayout = QVBoxLayout()
        mainVLayout.addLayout(labelLayout)
        self._progressBar = QProgressBar()
        self._progressBar.setRange(0, 100)
        mainVLayout.addWidget(self._progressBar)
        mainVLayout.addLayout(buttonLayout)
        self._detailsText = QPlainTextEdit()
        self._detailsText.setSizePolicy(QSizePolicy.MinimumExpanding,
                                        QSizePolicy.MinimumExpanding)
        self._detailsText.setReadOnly(True)
        self._detailsText.hide()
        mainVLayout.addWidget(self._detailsText, 1)
        mainVLayout.addStretch(0)
        self.setLayout(mainVLayout)

        #self.setFixedSize(320, self.sizeHint().height())
        #self.setFixedWidth(320)

    def changeEvent(self, event):
        if event.type() == QEvent.LanguageChange:
            self._cancelButton.setText(
                QCoreApplication.translate('QProgressDialogEx', 'Cancel')
            )
            self._detailsButton.setText(
                QCoreApplication.translate('QProgressDialogEx', 'Details')
            )
        else:
            QWidget.changeEvent(self, event)

    def setShowDetails(self, state):
        self._detailsText.setVisible(state)
        self.adjustSize()

    def _showDetails(self):
        self._detailsText.setVisible(not self._detailsText.isVisible())
        self.adjustSize()

    def _buttonClicked(self):
        if self.value() >= self.maximum():
            self.accept()
        else:
            self.cancel()

    def showDetails(self):
        self._detailsText.show()
        self.adjustSize()

    def hideDetails(self):
        self._detailsText.hide()
        self.adjustSize()

    def cancelButton(self):
        return self._cancelButton

    def detailsButton(self):
        return self._detailsButton

    def setIconPixmap(self, pixmap):
        self._pixmapLabel.setPixmap(pixmap)

    def accept(self):
        self.setValue(100)
        QDialog.accept(self)

    def cancel(self):
        self.canceled.emit()
        QDialog.reject(self)

    def reset(self):
        self.setResult(0)
        self._label.clear()
        self._detailsText.clear()
        self._detailsText.hide()
        self.adjustSize()
        self._progressBar.reset()
        self._cancelButton.setText(
            QCoreApplication.translate('QProgressDialogEx', 'Cancel'))
        self.setVisible(not self._autoclose)

    def autoClose(self):
        return self._autoclose

    def setAutoclose(self, state):
        self._autoclose = bool(state)

    def value(self):
        return self._progressBar.value()

    def setMinimum(self, value):
        return self._progressBar.setMinimum(value)

    def minimum(self):
        return self._progressBar.minimum()

    def setMaximum(self, value):
        return self._progressBar.setMaximum(value)

    def maximum(self):
        return self._progressBar.maximum()

    def setRange(self, minimum, maximum):
        return self._progressBar.setRange(minimum, maximum)

    def setValue(self, value):
        self._progressBar.setValue(value)
        if value >= self._progressBar.maximum():
            if self._autoclose:
                QDialog.accept(self)
            self._cancelButton.setText(
                QCoreApplication.translate('QProgressDialogEx', 'Close'))

    def labelText(self):
        return self._label.text()

    def setLabelText(self, text):
        return self._label.setText(text)

    def appendText(self, text):
        self._detailsText.appendPlainText(text)

    def exec(self):
        QDialog.exec(self)
        return self._progressBar.value()

if __name__ == '__main__':
    # pylupdate5 -verbose test.pro
    # add translations in "Qt Linguist"
    # lrelease test_si.tr
    '''
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    translator_si = QTranslator()
    translator_en = QTranslator()
    translator_si.load('testtr_si.qm')
    translator_en.load('testtr_en.qm')
    app.installTranslator(translator_si)

    xdlg = QTaskProgressDialog(
        None,
        QCoreApplication.translate("main", 'Test dialog'),
        QCoreApplication.translate("main", 'Some task state\nTest'),
        QPixmap('hgarsource_on').scaledToWidth(48)
    )
    xdlg.canceled.connect(lambda: print('Dialog canceled.'))
    xdlg.cancelButton().setIcon(QIcon(QPixmap('error.png')))
    xdlg.setHelp('Dialog help string.')

    xdlg1 = QProgressDialogEx(
        None,
        QCoreApplication.translate("main", 'Test dialog'),
        QCoreApplication.translate("main", 'Some task state\nTest'),
        QPixmap('hgarsource_on').scaledToWidth(48)
    )

    def update(dlg, xdlg1, tim):
        dlg.setValue(xdlg.value() + 1)
        if dlg.value() % 2:
            dlg.appendItem(''.join(('Task ', str(dlg.value()), ' done!',)))
        else:
            dlg.appendItem(''.join(('Task ', str(dlg.value()), ' done!',)),
                           QTaskProgressDialog.ERROR)
        xdlg1.appendText('One item')
        xdlg1.setValue(xdlg1.value() + 1)
        if dlg.value() >= dlg.maximum():
            tim.stop()

    tim = QTimer()
    tim.timeout.connect(lambda: update(xdlg, xdlg1, tim))
    tim.start(100)

    print('Result xdlg', xdlg.exec())
    print('Result xdlg1', xdlg1.exec())

    sys.exit(app.exec_())
    '''
    pass