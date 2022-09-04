"""
这个文件要实现UI的回调，处理用户输入，以及启动异常检测程序。
"""

from UserInterface import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMessageBox,
    QFileDialog,
    QPushButton,
    QLineEdit,
    QLabel,
    QProgressBar,
    QCheckBox,
)
from OutlierDetect import RunEvaluator, DetectionConfig, MODEL_ZOO
import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("[Application]")


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.cbModelName.addItems(MODEL_ZOO.model_list)
        self.ui.pgbEvaluator.reset()

    @pyqtSlot(str)
    def on_cbModelName_currentTextChanged(self, value: str):
        LOG.info(f'cbModelName: {value}')

    @pyqtSlot(str)
    def on_leNumTrain_textChanged(self, value: str):
        LOG.info(f'leNumTrain: {value}')

    @pyqtSlot(str)
    def on_leNumTest_textChanged(self, value: str):
        LOG.info(f'leNumTest: {value}')

    @pyqtSlot(str)
    def on_leOutlierRate_textChanged(self, value: str):
        LOG.info(f'leOutlierRate: {value}')

    @pyqtSlot()
    def on_pbRunDetect_clicked(self):
        LOG.info(f'pbRunDetect clicked')
        # job = RunEvaluator(
        #     parent=self,
        #     config=DetectionConfig(
        #         model_name="KNN",
        #         contamination=0.1,
        #         n_train=200,
        #         n_test=100,
        #     ),
        #     slot_dict={
        #         key: self.default_slot for key in RunEvaluator.ACTION_LIST}
        # )
        # job.start()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
