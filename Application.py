"""
这个文件要实现UI的回调，处理用户输入，以及启动异常检测程序。
"""

from UserInterface import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QMessageBox,
    QFileDialog,
    QPushButton,
    QLineEdit,
    QLabel,
    QProgressBar,
    QCheckBox,
)


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    # 槽函数的命名：on_ObjectName_SignalName
    @pyqtSlot()
    def on_pbDemo_clicked(self):
        QMessageBox.information(
            self, "欢迎来到PyQt", "121212212", QMessageBox.StandardButton.Yes
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
