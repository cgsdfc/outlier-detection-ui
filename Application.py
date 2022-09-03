"""
这个文件要实现UI的回调，处理用户输入，以及启动异常检测程序。
"""

from UserInterface import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QThreadPool, QRunnable, QObject
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
from OutlierDetect import RunEvaluator, DetectionConfig


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.thread_pool = QThreadPool(self)
        # self.thread_pool.setMaxThreadCount(4)

    @pyqtSlot(str)
    def default_slot(self, msg: str):
        print(f'XXXXXXXXXXXXXXXX {msg}')

    @pyqtSlot()
    def on_pbDemo_clicked(self):
        job = RunEvaluator(
            parent=self,
            config=DetectionConfig(
                model_name="KNN",
                contamination=0.1,
                n_train=200,
                n_test=100,
            ),
            slot_dict={
                key: self.default_slot for key in RunEvaluator.ACTION_LIST}
        )
        # self.thread_pool.start(job)
        job.start()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
