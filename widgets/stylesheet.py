# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:34:37 2017

@author: miran
"""
from widgets import ICON_PATH

TEMPLATE = \
'''
QPushButton {
    icon-size: $ICON_WIDTH$, $ICON_HEIGHT$;
    height: $ITEM_HEIGHT$;
}

QCheckablePushButton{
    width: $ITEM_HEIGHT$;
}

QShowHidePushButton{
    width: $ITEM_HEIGHT$;
}

QDrawErasePushButton{
    width: $ITEM_HEIGHT$;
}

QLockUnlockPushButton{
    width: $ITEM_HEIGHT$;
}

QColorSelectionButton{
   width: $ITEM_HEIGHT$;
}

QSquarePushButton{
   width: $ITEM_HEIGHT$;
}

QThinPushButton{
   icon-size: $SMALL_ICON_WIDTH$, $SMALL_ICON_HEIGHT$;
   width: $SMALL_ITEM_WIDTH$;
}

QMessageBox {
    dialogbuttonbox-buttons-have-icons: true;
    dialog-ok-icon: url($ICON_PATH$/ok.png);
    dialog-cancel-icon: url($ICON_PATH$/cancel.png);
}

QInputDialog {
    dialogbuttonbox-buttons-have-icons: true;
    dialog-ok-icon: url($ICON_PATH$/ok.png);
    dialog-cancel-icon: url($ICON_PATH$/cancel.png);
}

QSlider {
    height: $ITEM_HEIGHT$;
}

QSpinBox{
    height: $ITEM_HEIGHT$;
}

QLineEdit{
    height: $ITEM_HEIGHT$;
}

QListWidget::item:selected{
    border-style: solid;
    background-color: #e0e0e0;
    border-width: 2px;
    border-color: darkgray;
}

QDoubleSpinBox{
    height: $ITEM_HEIGHT$;
}

QComboBox{
    height: $ITEM_HEIGHT$;
}

QComboBox QAbstractItemView::item{
    min-height: $ITEM_HEIGHT$;
}

QClickableLabel{
    height: $ITEM_HEIGHT$;
}

QHeaderView::section {
    background-color: lightgray;
}

QSplitter::handle
{
    background-color: darkgray;
}

QToolBar
{
    icon-size: $ICON_WIDTH$, $ICON_HEIGHT$;
    height: $ITEM_HEIGHT$;
}
QToolButton{
    height: $ITEM_HEIGHT$;
}

QTabBar::tab
{
     height: $ITEM_HEIGHT$;
}
'''

'''
QCheckBox::indicator {
     width: $ITEM_HEIGHT$;
     height: $ITEM_HEIGHT$;
     spacing: 1;
}
'''


def assignValues(valuedict, template):
    for key in valuedict:
        template = template.replace('${}$'.format(key), str(valuedict[key]))
    return template

def defaultStyleSheet():
    return assignValues(
        {
            'ICON_PATH':ICON_PATH.replace('\\', '/'),
            'ITEM_WIDTH':'30px',
            'ITEM_HEIGHT':'30px',
            'ICON_HEIGHT':'25px',
            'ICON_WIDTH':'25px',
            'SMALL_ITEM_WIDTH':'15px',
            'SMALL_ITEM_HEIGHT':'15px',
            'SMALL_ICON_WIDTH':'14px',
            'SMALL_ICON_HEIGHT':'14px'
        },
        TEMPLATE
        )

def defaultDialogIconWidth():
    return 48
