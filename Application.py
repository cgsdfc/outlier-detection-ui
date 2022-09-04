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
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from OutlierDetect import RunEvaluator, DetectionConfig, MODEL_ZOO
import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("[Application]")


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.detection_config = DetectionConfig()

        self.ui.cbModelName.addItems(MODEL_ZOO.model_list)
        self.ui.pgbEvaluator.reset()
        # self.ui.leNumTrain.setValidator(QIntValidator(1, 1000, self))
        # self.ui.leNumTest.setValidator(QIntValidator(0, 500, self))
        # self.ui.leOutlierRate.setValidator(QDoubleValidator(0.1, 0.5, 2, self))

    @pyqtSlot(str)
    def on_cbModelName_currentTextChanged(self, value: str):
        LOG.info(f'cbModelName: {value}')
        self.detection_config.model_name = value

    @pyqtSlot()
    def on_leNumTrain_editingFinished(self):
        value = self.ui.leNumTrain.text()
        LOG.info(f'leNumTrain: {value}')
        try:
            self.detection_config.n_train = int(value)
        except ValueError:
            pass


    @pyqtSlot()
    def on_leNumTest_editingFinished(self):
        value = self.ui.leNumTest.text()
        LOG.info(f'leNumTest: {value}')
        try:
            self.detection_config.n_test = int(value)
        except ValueError:
            pass

    @pyqtSlot()
    def on_leOutlierRate_editingFinished(self):
        value = self.ui.leOutlierRate.text()
        LOG.info(f'leOutlierRate: {value}')
        self.detection_config.contamination = float(value)

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
