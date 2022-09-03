# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UserInterface.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(825, 590)
        MainWindow.setStyleSheet("/*  ------------------------------------------------------------------------  */\n"
"/* QtMaterial - https://github.com/UN-GCPDS/qt-material\n"
"/* By Yeison Cardona - GCPDS\n"
"/*  ------------------------------------------------------------------------  */\n"
"\n"
"*{\n"
"  color: #555555;\n"
"\n"
"  font-family: Roboto;\n"
"\n"
"  \n"
"    font-size: 13px;\n"
"  \n"
"\n"
"  \n"
"    line-height: 13px;\n"
"  \n"
"\n"
"  selection-background-color: #75a7ff;\n"
"  selection-color: #3c3c3c;\n"
"}\n"
"\n"
"*:focus {\n"
"   outline: none;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Custom colors  */\n"
"\n"
".danger{\n"
"  color: #dc3545;\n"
"  background-color: transparent;\n"
"}\n"
"\n"
".warning{\n"
"  color: #ffc107;\n"
"  background-color: transparent;\n"
"}\n"
"\n"
".success{\n"
"  color: #17a2b8;\n"
"  background-color: transparent;\n"
"}\n"
"\n"
".danger:disabled{\n"
"  color: rgba(220, 53, 69, 0.4);\n"
"  border-color: rgba(220, 53, 69, 0.4);\n"
"}\n"
"\n"
".warning:disabled{\n"
"  color: rgba(255, 193, 7, 0.4);\n"
"  border-color: rgba(255, 193, 7, 0.4);\n"
"}\n"
"\n"
".success:disabled{\n"
"  color: rgba(23, 162, 184, 0.4);\n"
"  border-color: rgba(23, 162, 184, 0.4);\n"
"}\n"
"\n"
".danger:flat:disabled{\n"
"  background-color: rgba(220, 53, 69, 0.1);\n"
"}\n"
"\n"
".warning:flat:disabled{\n"
"  background-color: rgba(255, 193, 7, 0.1);\n"
"}\n"
"\n"
".success:flat:disabled{\n"
"  background-color: rgba(23, 162, 184, 0.1);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Basic widgets  */\n"
"\n"
"QWidget {\n"
"  background-color: #e6e6e6;\n"
"}\n"
"\n"
"QGroupBox,\n"
"QFrame {\n"
"  background-color: #e6e6e6;\n"
"  border: 2px solid #ffffff;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QGroupBox.fill_background,\n"
"QFrame.fill_background {\n"
"  background-color: #f5f5f5;\n"
"  border: 2px solid #f5f5f5;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QSplitter {\n"
"  background-color: transparent;\n"
"  border: none\n"
"}\n"
"\n"
"QStatusBar {\n"
"  color: #555555;\n"
"  background-color: rgba(255, 255, 255, 0.2);\n"
"  border-radius: 0px;\n"
"}\n"
"\n"
"QScrollArea,\n"
"QStackedWidget,\n"
"QWidget > QToolBox,\n"
"QToolBox > QWidget,\n"
"QTabWidget > QWidget {\n"
"  border: none;\n"
"}\n"
"\n"
"QTabWidget::pane {\n"
"  border: none;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Inputs  */\n"
"\n"
"QDateTimeEdit,\n"
"QSpinBox,\n"
"QDoubleSpinBox,\n"
"QTextEdit,\n"
"QLineEdit,\n"
"QPushButton {\n"
"  color: #2979ff;\n"
"  background-color: #e6e6e6;\n"
"  border: 2px solid #2979ff;\n"
"  border-radius: 4px;\n"
"  height: 32px;\n"
"}\n"
"\n"
"QDateTimeEdit,\n"
"QSpinBox,\n"
"QDoubleSpinBox,\n"
"QTreeView,\n"
"QListView,\n"
"QLineEdit,\n"
"QComboBox {\n"
"  padding-left: 16px;\n"
"  border-radius: 0px;\n"
"  background-color: #f5f5f5;\n"
"  border-width: 0 0 2px 0;\n"
"  border-radius: 0px;\n"
"  border-top-left-radius: 4px;\n"
"  border-top-right-radius: 4px;\n"
"  height: 32px;\n"
"}\n"
"\n"
"QPlainTextEdit {\n"
"  border-radius: 4px;\n"
"  padding: 8px 16px;\n"
"  background-color: #e6e6e6;\n"
"  border: 2px solid #ffffff;\n"
"}\n"
"\n"
"QTextEdit {\n"
"  padding: 8px 16px;\n"
"  border-radius: 4px;\n"
"  background-color: #f5f5f5;\n"
"}\n"
"\n"
"QDateTimeEdit:disabled,\n"
"QSpinBox:disabled,\n"
"QDoubleSpinBox:disabled,\n"
"QTextEdit:disabled,\n"
"QLineEdit:disabled {\n"
"  color: rgba(41, 121, 255, 0.2);\n"
"  background-color: rgba(245, 245, 245, 0.75);\n"
"  border: 2px solid rgba(41, 121, 255, 0.2);\n"
"  border-width: 0 0 2px 0;\n"
"  padding: 0px 16px;\n"
"  border-radius: 0px;\n"
"  border-top-left-radius: 4px;\n"
"  border-top-right-radius: 4px;\n"
"  height: 32px;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QComboBox  */\n"
"\n"
"QComboBox {\n"
"  color: #2979ff;\n"
"  border: 1px solid #2979ff;\n"
"  border-width: 0 0 2px 0;\n"
"  background-color: #f5f5f5;\n"
"  border-radius: 0px;\n"
"  border-top-left-radius: 4px;\n"
"  border-top-right-radius: 4px;\n"
"  height: 32px;\n"
"}\n"
"\n"
"QComboBox:disabled {\n"
"  color: rgba(41, 121, 255, 0.2);\n"
"  background-color: rgba(245, 245, 245, 0.75);\n"
"  border-bottom: 2px solid rgba(41, 121, 255, 0.2);\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"  border: none;\n"
"  color: #2979ff;\n"
"  width: 20px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"  image: url(icon:/primary/downarrow.svg);\n"
"  margin-right: 12px;\n"
"}\n"
"\n"
"QComboBox::down-arrow:disabled {\n"
"  image: url(icon:/disabled/downarrow.svg);\n"
"  margin-right: 12px;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"  background-color: #f5f5f5;\n"
"  border: 2px solid #ffffff;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QComboBox[frame=\'false\'] {\n"
"  color: #2979ff;\n"
"  background-color: transparent;\n"
"  border: 1px solid transparent;\n"
"}\n"
"QComboBox[frame=\'false\']:disabled {\n"
"  color: rgba(41, 121, 255, 0.2);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Spin buttons  */\n"
"\n"
"QDateTimeEdit::up-button,\n"
"QDoubleSpinBox::up-button,\n"
"QSpinBox::up-button {\n"
"  subcontrol-origin: border;\n"
"  subcontrol-position: top right;\n"
"  width: 20px;\n"
"  image: url(icon:/primary/uparrow.svg);\n"
"  border-width: 0px;\n"
"  margin-right: 5px;\n"
"}\n"
"\n"
"QDateTimeEdit::up-button:disabled,\n"
"QDoubleSpinBox::up-button:disabled,\n"
"QSpinBox::up-button:disabled {\n"
"  image: url(icon:/disabled/uparrow.svg);\n"
"}\n"
"\n"
"QDateTimeEdit::down-button,\n"
"QDoubleSpinBox::down-button,\n"
"QSpinBox::down-button {\n"
"  subcontrol-origin: border;\n"
"  subcontrol-position: bottom right;\n"
"  width: 20px;\n"
"  image: url(icon:/primary/downarrow.svg);\n"
"  border-width: 0px;\n"
"  border-top-width: 0;\n"
"  margin-right: 5px;\n"
"}\n"
"\n"
"QDateTimeEdit::down-button:disabled,\n"
"QDoubleSpinBox::down-button:disabled,\n"
"QSpinBox::down-button:disabled {\n"
"  image: url(icon:/disabled/downarrow.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QPushButton  */\n"
"\n"
"QPushButton {\n"
"  text-transform: uppercase;\n"
"  margin: 0px;\n"
"  padding: 1px 16px;\n"
"  height: 32px;\n"
"  font-weight: bold;\n"
"\n"
"  \n"
"    border-radius: 4px;\n"
"  \n"
"\n"
"\n"
"}\n"
"\n"
"QPushButton:checked,\n"
"QPushButton:pressed {\n"
"  color: #e6e6e6;\n"
"  background-color: #2979ff;\n"
"}\n"
"\n"
"QPushButton:flat {\n"
"  margin: 0px;\n"
"  color: #2979ff;\n"
"  border: none;\n"
"  background-color: transparent;\n"
"}\n"
"\n"
"QPushButton:flat:hover {\n"
"  background-color: rgba(41, 121, 255, 0.2);\n"
"}\n"
"\n"
"QPushButton:flat:pressed,\n"
"QPushButton:flat:checked {\n"
"  background-color: rgba(41, 121, 255, 0.1);\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"  color: rgba(255, 255, 255, 0.75);\n"
"  background-color: transparent;\n"
"  border-color:  #ffffff;\n"
"}\n"
"\n"
"QPushButton:flat:disabled {\n"
"  color: rgba(255, 255, 255, 0.75);\n"
"  background-color: rgba(255, 255, 255, 0.25);\n"
"  border: none;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"  border: 2px solid rgba(255, 255, 255, 0.75);\n"
"}\n"
"\n"
"QPushButton:checked:disabled {\n"
"  color: #f5f5f5;\n"
"  background-color: #ffffff;\n"
"  border-color:  #ffffff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QTabBar  */\n"
"\n"
"QTabBar{\n"
"  text-transform: uppercase;\n"
"  font-weight: bold;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"  color: #555555;\n"
"  border: 0px;\n"
"}\n"
"\n"
"QTabBar::tab:bottom,\n"
"QTabBar::tab:top{\n"
"  padding: 0 16px;\n"
"  height: 28px;\n"
"}\n"
"\n"
"QTabBar::tab:left,\n"
"QTabBar::tab:right{\n"
"  padding: 16px 0;\n"
"  width: 28px;\n"
"}\n"
"\n"
"QTabBar::tab:top:selected,\n"
"QTabBar::tab:top:hover {\n"
"  color: #2979ff;\n"
"  border-bottom: 2px solid #2979ff;\n"
"}\n"
"\n"
"QTabBar::tab:bottom:selected,\n"
"QTabBar::tab:bottom:hover {\n"
"  color: #2979ff;\n"
"  border-top: 2px solid #2979ff;\n"
"}\n"
"\n"
"QTabBar::tab:right:selected,\n"
"QTabBar::tab:right:hover {\n"
"  color: #2979ff;\n"
"  border-left: 2px solid #2979ff;\n"
"}\n"
"\n"
"QTabBar::tab:left:selected,\n"
"QTabBar::tab:left:hover {\n"
"  color: #2979ff;\n"
"  border-right: 2px solid #2979ff;\n"
"}\n"
"\n"
"QTabBar QToolButton:hover,\n"
"QTabBar QToolButton {\n"
"  border: 20px;\n"
"  background-color: #e6e6e6;\n"
"}\n"
"\n"
"QTabBar QToolButton::up-arrow {\n"
"  image: url(icon:/disabled/uparrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::up-arrow:hover {\n"
"  image: url(icon:/primary/uparrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::down-arrow {\n"
"  image: url(icon:/disabled/downarrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::down-arrow:hover {\n"
"  image: url(icon:/primary/downarrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::right-arrow {\n"
"  image: url(icon:/primary/rightarrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::right-arrow:hover {\n"
"  image: url(icon:/disabled/rightarrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::left-arrow {\n"
"  image: url(icon:/primary/leftarrow2.svg);\n"
"}\n"
"\n"
"QTabBar QToolButton::left-arrow:hover {\n"
"  image: url(icon:/disabled/leftarrow2.svg);\n"
"}\n"
"\n"
"QTabBar::close-button {\n"
"  image: url(icon:/disabled/tab_close.svg);\n"
"}\n"
"\n"
"QTabBar::close-button:hover {\n"
"  image: url(icon:/primary/tab_close.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QGroupBox  */\n"
"\n"
"QGroupBox {\n"
"  padding: 16px;\n"
"  padding-top: 36px;\n"
"  line-height: ;\n"
"  text-transform: uppercase;\n"
"  font-size: ;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"  color: rgba(85, 85, 85, 0.4);\n"
"  subcontrol-origin: margin;\n"
"  subcontrol-position: top left;\n"
"  padding: 16px;\n"
"  background-color: #e6e6e6;\n"
"  background-color: transparent;\n"
"  height: 36px;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QRadioButton and QCheckBox labels  */\n"
"\n"
"QRadioButton,\n"
"QCheckBox {\n"
"  spacing: 12px;\n"
"  color: #555555;\n"
"  line-height: 14px;\n"
"  height: 36px;\n"
"  background-color: transparent;\n"
"  spacing: 5px;\n"
"}\n"
"\n"
"QRadioButton:disabled,\n"
"QCheckBox:disabled {\n"
"  color: rgba(85, 85, 85, 0.3);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  General Indicators  */\n"
"\n"
"QGroupBox::indicator {\n"
"  width: 24px;\n"
"  height: 24px;\n"
"  border-radius: 3px;\n"
"}\n"
"\n"
"QMenu::indicator,\n"
"QListView::indicator,\n"
"QTableWidget::indicator,\n"
"QRadioButton::indicator,\n"
"QCheckBox::indicator {\n"
"  width: 28px;\n"
"  height: 28px;\n"
"  border-radius: 4px;\n"
" }\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QListView Indicator  */\n"
"\n"
"QListView::indicator:checked,\n"
"QListView::indicator:checked:selected,\n"
"QListView::indicator:checked:focus {\n"
"  image: url(icon:/primary/checklist.svg);\n"
"}\n"
"\n"
"QListView::indicator:checked:selected:active {\n"
"  image: url(icon:/primary/checklist_invert.svg);\n"
"}\n"
"\n"
"QListView::indicator:checked:disabled {\n"
"  image: url(icon:/disabled/checklist.svg);\n"
"}\n"
"\n"
"QListView::indicator:indeterminate,\n"
"QListView::indicator:indeterminate:selected,\n"
"QListView::indicator:indeterminate:focus {\n"
"  image: url(icon:/primary/checklist_indeterminate.svg);\n"
"}\n"
"\n"
"QListView::indicator:indeterminate:selected:active {\n"
"  image: url(icon:/primary/checklist_indeterminate_invert.svg);\n"
"}\n"
"\n"
"QListView::indicator:indeterminate:disabled {\n"
"  image: url(icon:/disabled/checklist_indeterminate.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QTableView Indicator  */\n"
"\n"
"QTableView::indicator:enabled:checked,\n"
"QTableView::indicator:enabled:checked:selected,\n"
"QTableView::indicator:enabled:checked:focus {\n"
"  image: url(icon:/primary/checkbox_checked.svg);\n"
"}\n"
"\n"
"QTableView::indicator:checked:selected:active {\n"
"  image: url(icon:/primary/checkbox_checked_invert.svg);\n"
"}\n"
"\n"
"QTableView::indicator:disabled:checked,\n"
"QTableView::indicator:disabled:checked:selected,\n"
"QTableView::indicator:disabled:checked:focus {\n"
"  image: url(icon:/disabled/checkbox_checked.svg);\n"
"}\n"
"\n"
"QTableView::indicator:enabled:unchecked,\n"
"QTableView::indicator:enabled:unchecked:selected,\n"
"QTableView::indicator:enabled:unchecked:focus {\n"
"  image: url(icon:/primary/checkbox_unchecked.svg);\n"
"}\n"
"\n"
"QTableView::indicator:unchecked:selected:active {\n"
"  image: url(icon:/primary/checkbox_unchecked_invert.svg);\n"
"}\n"
"\n"
"QTableView::indicator:disabled:unchecked,\n"
"QTableView::indicator:disabled:unchecked:selected,\n"
"QTableView::indicator:disabled:unchecked:focus {\n"
"  image: url(icon:/disabled/checkbox_unchecked.svg);\n"
"}\n"
"\n"
"QTableView::indicator:enabled:indeterminate,\n"
"QTableView::indicator:enabled:indeterminate:selected,\n"
"QTableView::indicator:enabled:indeterminate:focus {\n"
"  image: url(icon:/primary/checkbox_indeterminate.svg);\n"
"}\n"
"\n"
"QTableView::indicator:indeterminate:selected:active {\n"
"  image: url(icon:/primary/checkbox_indeterminate_invert.svg);\n"
"}\n"
"\n"
"QTableView::indicator:disabled:indeterminate,\n"
"QTableView::indicator:disabled:indeterminate:selected,\n"
"QTableView::indicator:disabled:indeterminate:focus {\n"
"  image: url(icon:/disabled/checkbox_indeterminate.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QCheckBox and QGroupBox Indicator  */\n"
"\n"
"QCheckBox::indicator:checked,\n"
"QGroupBox::indicator:checked {\n"
"  image: url(icon:/primary/checkbox_checked.svg);\n"
"}\n"
"\n"
"QCheckBox::indicator:unchecked,\n"
"QGroupBox::indicator:unchecked {\n"
"  image: url(icon:/primary/checkbox_unchecked.svg);\n"
"}\n"
"\n"
"QCheckBox::indicator:indeterminate,\n"
"QGroupBox::indicator:indeterminate {\n"
"  image: url(icon:/primary/checkbox_indeterminate.svg);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked:disabled,\n"
"QGroupBox::indicator:checked:disabled {\n"
"  image: url(icon:/disabled/checkbox_checked.svg);\n"
"}\n"
"\n"
"QCheckBox::indicator:unchecked:disabled,\n"
"QGroupBox::indicator:unchecked:disabled {\n"
"  image: url(icon:/disabled/checkbox_unchecked.svg);\n"
"}\n"
"\n"
"QCheckBox::indicator:indeterminate:disabled,\n"
"QGroupBox::indicator:indeterminate:disabled {\n"
"  image: url(icon:/disabled/checkbox_indeterminate.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QRadioButton Indicator  */\n"
"\n"
"QRadioButton::indicator:checked {\n"
"  image: url(icon:/primary/radiobutton_checked.svg);\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked {\n"
"  image: url(icon:/primary/radiobutton_unchecked.svg);\n"
"}\n"
"\n"
"QRadioButton::indicator:checked:disabled {\n"
"  image: url(icon:/disabled/radiobutton_checked.svg);\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:disabled {\n"
"  image: url(icon:/disabled/radiobutton_unchecked.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QDockWidget  */\n"
"\n"
"QDockWidget {\n"
"  color: #555555;\n"
"  text-transform: uppercase;\n"
"  border: 2px solid #f5f5f5;\n"
"  titlebar-close-icon: url(icon:/primary/close.svg);\n"
"  titlebar-normal-icon: url(icon:/primary/float.svg);\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QDockWidget::title {\n"
"  text-align: left;\n"
"  padding-left: 36px;\n"
"  padding: 3px;\n"
"  margin-top: 4px;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QComboBox indicator  */\n"
"\n"
"QComboBox::indicator:checked {\n"
"  image: url(icon:/primary/checklist.svg);\n"
"}\n"
"\n"
"QComboBox::indicator:checked:selected {\n"
"  image: url(icon:/primary/checklist_invert.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Menu Items  */\n"
"\n"
"QComboBox::item,\n"
"QCalendarWidget QMenu::item,\n"
"QMenu::item {\n"
"  \n"
"    height: 28px;\n"
"  \n"
"  border: 8px solid transparent;\n"
"  color: #555555;\n"
"}\n"
"\n"
"QCalendarWidget QMenu::item,\n"
"QMenu::item {\n"
"  \n"
"    \n"
"  \n"
"}\n"
"\n"
"\n"
"QComboBox::item:selected,\n"
"QCalendarWidget QMenu::item:selected,\n"
"QMenu::item:selected {\n"
"  color: #3c3c3c;\n"
"  background-color: #75a7ff;\n"
"  border-radius: 0px;\n"
"}\n"
"\n"
"QComboBox::item:disabled,\n"
"QCalendarWidget QMenu::item:disabled,\n"
"QMenu::item:disabled {\n"
"  color: rgba(85, 85, 85, 0.3);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QMenu  */\n"
"\n"
"QCalendarWidget QMenu,\n"
"QMenu {\n"
"  background-color: #f5f5f5;\n"
"  border: 2px solid #ffffff;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"  height: 2px;\n"
"  background-color: #ffffff;\n"
"  margin-left: 2px;\n"
"  margin-right: 2px;\n"
"}\n"
"\n"
"QMenu::right-arrow{\n"
"  image: url(icon:/primary/rightarrow.svg);\n"
"  width: 16px;\n"
"  height: 16px;\n"
"}\n"
"\n"
"QMenu::right-arrow:selected{\n"
"  image: url(icon:/disabled/rightarrow.svg);\n"
"}\n"
"\n"
"QMenu::indicator:non-exclusive:unchecked {\n"
"  image: url(icon:/primary/checkbox_unchecked.svg);\n"
"}\n"
"\n"
"QMenu::indicator:non-exclusive:unchecked:selected {\n"
"  image: url(icon:/primary/checkbox_unchecked_invert.svg);\n"
"}\n"
"\n"
"QMenu::indicator:non-exclusive:checked {\n"
"  image: url(icon:/primary/checkbox_checked.svg);\n"
"}\n"
"\n"
"QMenu::indicator:non-exclusive:checked:selected {\n"
"  image: url(icon:/primary/checkbox_checked_invert.svg);\n"
"}\n"
"\n"
"QMenu::indicator:exclusive:unchecked {\n"
"  image: url(icon:/primary/radiobutton_unchecked.svg);\n"
"}\n"
"\n"
"QMenu::indicator:exclusive:unchecked:selected {\n"
"  image: url(icon:/primary/radiobutton_unchecked_invert.svg);\n"
"}\n"
"\n"
"QMenu::indicator:exclusive:checked {\n"
"  image: url(icon:/primary/radiobutton_checked.svg);\n"
"}\n"
"\n"
"QMenu::indicator:exclusive:checked:selected {\n"
"  image: url(icon:/primary/radiobutton_checked_invert.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QMenuBar  */\n"
"\n"
"QMenuBar {\n"
"  background-color: #f5f5f5;\n"
"  color: #555555;\n"
"}\n"
"\n"
"QMenuBar::item {\n"
"  height: 32px;\n"
"  padding: 8px;\n"
"  background-color: transparent;\n"
"  color: #555555;\n"
"}\n"
"\n"
"QMenuBar::item:selected,\n"
"QMenuBar::item:pressed {\n"
"  color: #3c3c3c;\n"
"  background-color: #75a7ff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QToolBox  */\n"
"\n"
"QToolBox::tab {\n"
"  background-color: #f5f5f5;\n"
"  color: #555555;\n"
"  text-transform: uppercase;\n"
"  border-radius: 4px;\n"
"  padding-left: 15px;\n"
"}\n"
"\n"
"QToolBox::tab:selected,\n"
"QToolBox::tab:hover {\n"
"  background-color: rgba(41, 121, 255, 0.2);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QProgressBar  */\n"
"\n"
"QProgressBar {\n"
"  border-radius: 0;\n"
"  background-color: #ffffff;\n"
"  text-align: center;\n"
"  color: transparent;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"  background-color: #2979ff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QScrollBar  */\n"
"\n"
"QScrollBar:horizontal {\n"
"  border: 0;\n"
"  background: #f5f5f5;\n"
"  height: 8px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"  border: 0;\n"
"  background: #f5f5f5;\n"
"  width: 8px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"  background: rgba(41, 121, 255, 0.1);\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"  min-width: 24px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"  min-height: 24px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical:hover,\n"
"QScrollBar::handle:horizontal:hover {\n"
"  background: #2979ff;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical,\n"
"QScrollBar::sub-line:vertical,\n"
"QScrollBar::add-line:horizontal,\n"
"QScrollBar::sub-line:horizontal {\n"
"  border: 0;\n"
"  background: transparent;\n"
"  width: 0px;\n"
"  height: 0px;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QScrollBar-Big  */\n"
"\n"
"QScrollBar.big:horizontal {\n"
"  border: 0;\n"
"  background: #f5f5f5;\n"
"  height: 36px;\n"
"}\n"
"\n"
"QScrollBar.big:vertical {\n"
"  border: 0;\n"
"  background: #f5f5f5;\n"
"  width: 36px;\n"
"}\n"
"\n"
"QScrollBar.big::handle,\n"
"QScrollBar.big::handle:vertical:hover,\n"
"QScrollBar.big::handle:horizontal:hover {\n"
"  background: #2979ff;\n"
"}\n"
"\n"
"QScrollBar.big::handle:horizontal {\n"
"  min-width: 24px;\n"
"}\n"
"\n"
"QScrollBar.big::handle:vertical {\n"
"  min-height: 24px;\n"
"}\n"
"\n"
"QScrollBar.big::add-line:vertical,\n"
"QScrollBar.big::sub-line:vertical,\n"
"QScrollBar.big::add-line:horizontal,\n"
"QScrollBar.big::sub-line:horizontal {\n"
"  border: 0;\n"
"  background: transparent;\n"
"  width: 0px;\n"
"  height: 0px;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QSlider  */\n"
"\n"
"QSlider:horizontal {\n"
"  min-height: 24px;\n"
"  max-height: 24px;\n"
"}\n"
"\n"
"QSlider:vertical {\n"
"  min-width: 24px;\n"
"  max-width: 24px;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"  height: 4px;\n"
"  background: #393939;\n"
"  margin: 0 12px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"  width: 4px;\n"
"  background: #393939;\n"
"  margin: 12px 0;\n"
"  border-radius: 24px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"  image: url(icon:/primary/slider.svg);\n"
"  width: 24px;\n"
"  height: 24px;\n"
"  margin: -24px -12px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"  image: url(icon:/primary/slider.svg);\n"
"  border-radius: 24px;\n"
"  width: 24px;\n"
"  height: 24px;\n"
"  margin: -12px -24px;\n"
"}\n"
"\n"
"QSlider::add-page {\n"
"background: #f5f5f5;\n"
"}\n"
"\n"
"QSlider::sub-page {\n"
"background: #2979ff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QLabel  */\n"
"\n"
"QLabel {\n"
"  border: none;\n"
"  background: transparent;\n"
"  color: #555555\n"
"}\n"
"\n"
"QLabel:disabled {\n"
"  color: rgba(85, 85, 85, 0.2)\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  VLines and HLinex  */\n"
"\n"
"QFrame[frameShape=\"4\"] {\n"
"    border-width: 1px 0 0 0;\n"
"    background: none;\n"
"}\n"
"\n"
"QFrame[frameShape=\"5\"] {\n"
"    border-width: 0 1px 0 0;\n"
"    background: none;\n"
"}\n"
"\n"
"QFrame[frameShape=\"4\"],\n"
"QFrame[frameShape=\"5\"] {\n"
"  border-color: #ffffff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QToolBar  */\n"
"\n"
"QToolBar {\n"
"  background: #e6e6e6;\n"
"  border: 0px solid;\n"
"}\n"
"\n"
"QToolBar:horizontal {\n"
"  border-bottom: 1px solid #ffffff;\n"
"}\n"
"\n"
"QToolBar:vertical {\n"
"  border-right: 1px solid #ffffff;\n"
"}\n"
"\n"
"QToolBar::handle:horizontal {\n"
"  image: url(icon:/primary/toolbar-handle-horizontal.svg);\n"
"}\n"
"\n"
"QToolBar::handle:vertical {\n"
"  image: url(icon:/primary/toolbar-handle-vertical.svg);\n"
"}\n"
"\n"
"QToolBar::separator:horizontal {\n"
"  border-right: 1px solid #ffffff;\n"
"  border-left: 1px solid #ffffff;\n"
"  width: 1px;\n"
"}\n"
"\n"
"QToolBar::separator:vertical {\n"
"  border-top: 1px solid #ffffff;\n"
"  border-bottom: 1px solid #ffffff;\n"
"  height: 1px;\n"
"}\n"
"\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QToolButton  */\n"
"\n"
"QToolButton {\n"
"  background: #e6e6e6;\n"
"  border: 0px;\n"
"  height: 36px;\n"
"  margin: 3px;\n"
"  padding: 3px;\n"
"  border-right: 12px solid #e6e6e6;\n"
"  border-left: 12px solid #e6e6e6;\n"
"}\n"
"\n"
"QToolButton:hover {\n"
"  background: #ffffff;\n"
"  border-right: 12px solid #ffffff;\n"
"  border-left: 12px solid #ffffff;\n"
"}\n"
"\n"
"QToolButton:pressed {\n"
"  background: #f5f5f5;\n"
"  border-right: 12px solid #f5f5f5;\n"
"  border-left: 12px solid #f5f5f5;\n"
"}\n"
"\n"
"QToolButton:checked {\n"
"  background: #ffffff;\n"
"  border-left: 12px solid #ffffff;\n"
"  border-right: 12px solid #2979ff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  General viewers  */\n"
"\n"
"QTableView {\n"
"  background-color: #e6e6e6;\n"
"  border: 1px solid #f5f5f5;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QTreeView,\n"
"QListView {\n"
"  border-radius: 4px;\n"
"  padding: 4px;\n"
"  margin: 0px;\n"
"  border: 0px;\n"
"}\n"
"\n"
"QTableView::item,\n"
"QTreeView::item,\n"
"QListView::item {\n"
"  padding: 4px;\n"
"  min-height: 32px;\n"
"  color: #555555;\n"
"  selection-color: #555555; /* For Windows */\n"
"  border-color: transparent;  /* Fix #34 */\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Items Selection */\n"
"\n"
"QTableView::item:selected,\n"
"QTreeView::item:selected,\n"
"QListView::item:selected {\n"
"  background-color: rgba(41, 121, 255, 0.2);\n"
"  selection-background-color: rgba(41, 121, 255, 0.2);\n"
"  color: #555555;\n"
"  selection-color: #555555; /* For Windows */\n"
"}\n"
"\n"
"QTableView::item:selected:focus,\n"
"QTreeView::item:selected:focus,\n"
"QListView::item:selected:focus {\n"
"  background-color: #2979ff;\n"
"  selection-background-color: #2979ff;\n"
"  color: #3c3c3c;\n"
"  selection-color: #3c3c3c; /* For Windows */\n"
"}\n"
"\n"
"QTableView {\n"
"  selection-background-color: rgba(41, 121, 255, 0.2);\n"
"}\n"
"\n"
"QTableView:focus {\n"
"  selection-background-color: #2979ff;\n"
"}\n"
"\n"
"QTableView::item:disabled {\n"
"  color: rgba(85, 85, 85, 0.3);\n"
"  selection-color: rgba(85, 85, 85, 0.3);\n"
"  background-color: #f5f5f5;\n"
"  selection-background-color: #f5f5f5;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QTreeView  */\n"
"\n"
"QTreeView::branch{\n"
"  background-color: #f5f5f5;\n"
"}\n"
"\n"
"QTreeView::branch:closed:has-children:has-siblings,\n"
"QTreeView::branch:closed:has-children:!has-siblings {\n"
"  image: url(icon:/primary/branch-closed.svg);\n"
"}\n"
"\n"
"QTreeView::branch:open:has-children:!has-siblings,\n"
"QTreeView::branch:open:has-children:has-siblings {\n"
"  image: url(icon:/primary/branch-open.svg);\n"
"}\n"
"\n"
"QTreeView::branch:has-siblings:!adjoins-item {\n"
"  border-image: url(icon:/disabled/vline.svg) 0;\n"
"}\n"
"\n"
"QTreeView::branch:has-siblings:adjoins-item {\n"
"    border-image: url(icon:/disabled/branch-more.svg) 0;\n"
"}\n"
"\n"
"QTreeView::branch:!has-children:!has-siblings:adjoins-item,\n"
"QTreeView::branch:has-children:!has-siblings:adjoins-item {\n"
"    border-image: url(icon:/disabled/branch-end.svg) 0;\n"
"}\n"
"\n"
"QTreeView QHeaderView::section {\n"
"  border: none;\n"
"}\n"
"\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Custom buttons  */\n"
"\n"
"QPushButton.danger {\n"
"  border-color: #dc3545;\n"
"  color: #dc3545;\n"
"}\n"
"\n"
"QPushButton.danger:checked,\n"
"QPushButton.danger:pressed {\n"
"  color: #e6e6e6;\n"
"  background-color: #dc3545;\n"
"}\n"
"\n"
"QPushButton.warning{\n"
"  border-color: #ffc107;\n"
"  color: #ffc107;\n"
"}\n"
"\n"
"QPushButton.warning:checked,\n"
"QPushButton.warning:pressed {\n"
"  color: #e6e6e6;\n"
"  background-color: #ffc107;\n"
"}\n"
"\n"
"QPushButton.success {\n"
"  border-color: #17a2b8;\n"
"  color: #17a2b8;\n"
"}\n"
"\n"
"QPushButton.success:checked,\n"
"QPushButton.success:pressed {\n"
"  color: #e6e6e6;\n"
"  background-color: #17a2b8;\n"
"}\n"
"\n"
"QPushButton.danger:flat:hover {\n"
"  background-color: rgba(220, 53, 69, 0.2);\n"
"}\n"
"\n"
"QPushButton.danger:flat:pressed,\n"
"QPushButton.danger:flat:checked {\n"
"  background-color: rgba(220, 53, 69, 0.1);\n"
"  color: #dc3545;\n"
"}\n"
"\n"
"QPushButton.warning:flat:hover {\n"
"  background-color: rgba(255, 193, 7, 0.2);\n"
"}\n"
"\n"
"QPushButton.warning:flat:pressed,\n"
"QPushButton.warning:flat:checked {\n"
"  background-color: rgba(255, 193, 7, 0.1);\n"
"  color: #ffc107;\n"
"}\n"
"\n"
"QPushButton.success:flat:hover {\n"
"  background-color: rgba(23, 162, 184, 0.2);\n"
"}\n"
"\n"
"QPushButton.success:flat:pressed,\n"
"QPushButton.success:flat:checked {\n"
"  background-color: rgba(23, 162, 184, 0.1);\n"
"  color: #17a2b8;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QTableView  */\n"
"\n"
"QTableCornerButton::section {\n"
"  background-color: #f5f5f5;\n"
"  border-radius: 0px;\n"
"  border-right: 1px solid;\n"
"  border-bottom: 1px solid;\n"
"  border-color: #e6e6e6;\n"
"}\n"
"\n"
"QTableView {\n"
"  alternate-background-color: rgba(245, 245, 245, 0.7);\n"
"}\n"
"\n"
"QHeaderView {\n"
"  border: none;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"  color: rgba(85, 85, 85, 0.7);\n"
"  text-transform: uppercase;\n"
"  background-color: #f5f5f5;\n"
"  padding: 0 24px;\n"
"  height: 36px;\n"
"  border-radius: 0px;\n"
"  border-right: 1px solid;\n"
"  border-bottom: 1px solid;\n"
"  border-color: #e6e6e6;\n"
"}\n"
"\n"
"QHeaderView::section:vertical {\n"
"\n"
"}\n"
"\n"
"QHeaderView::section:horizontal {\n"
"\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QLCDNumber  */\n"
"\n"
"QLCDNumber {\n"
"  color: #2979ff;\n"
"  background-color:rgba(41, 121, 255, 0.1);\n"
"  border: 1px solid rgba(41, 121, 255, 0.3);\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QCalendarWidget  */\n"
"\n"
"#qt_calendar_prevmonth {\n"
"  qproperty-icon: url(icon:/primary/leftarrow.svg);\n"
"}\n"
"\n"
"#qt_calendar_nextmonth {\n"
"  qproperty-icon: url(icon:/primary/rightarrow.svg);\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Inline QLineEdit  */\n"
"\n"
"QTreeView QLineEdit,\n"
"QTableView QLineEdit,\n"
"QListView QLineEdit {\n"
"  color: #555555;\n"
"  background-color: #f5f5f5;\n"
"  border: 1px solid unset;\n"
"  border-radius: unset;\n"
"  padding: unset;\n"
"  padding-left: unset;\n"
"  height: unset;\n"
"  border-width: unset;\n"
"  border-top-left-radius: unset;\n"
"  border-top-right-radius: unset;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QToolTip  */\n"
"\n"
"QToolTip {\n"
"  padding: 4px;\n"
"  border: 1px solid #e6e6e6;\n"
"  border-radius: 4px;\n"
"  color: #555555;\n"
"  background-color: #ffffff;\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  QDialog  */\n"
"\n"
"\n"
"\n"
"QDialog QToolButton:disabled {\n"
"  background-color: #f5f5f5;\n"
"  color: #555555\n"
"}\n"
"\n"
"/*  ------------------------------------------------------------------------  */\n"
"/*  Grips  */\n"
"\n"
"\n"
"QMainWindow::separator:vertical,\n"
"QSplitter::handle:horizontal {\n"
"  image: url(icon:/primary/splitter-horizontal.svg);\n"
"}\n"
"\n"
"QMainWindow::separator:horizontal,\n"
"QSplitter::handle:vertical {\n"
"  image: url(icon:/primary/splitter-vertical.svg);\n"
"}\n"
"\n"
"QSizeGrip {\n"
"  image: url(icon:/primary/sizegrip.svg);\n"
"  background-color: transparent;\n"
"}\n"
"\n"
"QMenuBar QToolButton:hover,\n"
"QMenuBar QToolButton:pressed,\n"
"QMenuBar QToolButton {\n"
"  border-width: 0;\n"
"  border-left: 10px;\n"
"  border-image: url(icon:/primary/rightarrow2.svg);\n"
"  background-color: transparent;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.frame_2 = QtWidgets.QFrame(self.groupBox)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lb_image = QtWidgets.QLabel(self.frame_2)
        self.lb_image.setText("")
        self.lb_image.setObjectName("lb_image")
        self.gridLayout_3.addWidget(self.lb_image, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.frame_2, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.groupBox_2)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.cb_ratio = QtWidgets.QComboBox(self.frame)
        self.cb_ratio.setObjectName("cb_ratio")
        self.gridLayout_4.addWidget(self.cb_ratio, 1, 3, 1, 1)
        self.cb_modelName = QtWidgets.QComboBox(self.frame)
        self.cb_modelName.setObjectName("cb_modelName")
        self.gridLayout_4.addWidget(self.cb_modelName, 1, 0, 1, 1)
        self.cb_numTest = QtWidgets.QComboBox(self.frame)
        self.cb_numTest.setObjectName("cb_numTest")
        self.gridLayout_4.addWidget(self.cb_numTest, 1, 2, 1, 1)
        self.cb_numTrain = QtWidgets.QComboBox(self.frame)
        self.cb_numTrain.setObjectName("cb_numTrain")
        self.gridLayout_4.addWidget(self.cb_numTrain, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 0, 1, 1, 1)
        self.gridLayout_4.setRowStretch(0, 2)
        self.gridLayout_4.setRowStretch(1, 3)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.frame)
        self.pb_runDetect = QtWidgets.QPushButton(self.groupBox_2)
        self.pb_runDetect.setObjectName("pb_runDetect")
        self.horizontalLayout.addWidget(self.pb_runDetect)
        self.horizontalLayout.setStretch(0, 8)
        self.horizontalLayout.setStretch(1, 1)
        self.gridLayout_7.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 3)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "异常检测演示程序"))
        self.groupBox.setTitle(_translate("MainWindow", "检测结果"))
        self.groupBox_2.setTitle(_translate("MainWindow", "输入参数"))
        self.label_3.setText(_translate("MainWindow", "测试样本数"))
        self.label.setText(_translate("MainWindow", "选择模型"))
        self.label_4.setText(_translate("MainWindow", "异常点比例"))
        self.label_2.setText(_translate("MainWindow", "训练样本数"))
        self.pb_runDetect.setText(_translate("MainWindow", "检测"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
