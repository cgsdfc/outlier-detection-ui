"""
这个文件要实现UI的回调，处理用户输入，以及启动异常检测程序。
"""

import traceback
from typing import Callable, Dict
from UserInterface import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread
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
from OutlierDetect import MODEL_ZOO, DataConfig, DetectionEvaluator, ModelConfig
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("[Application]")

# class RENAME:
#     _NAMES = dict(KNN='K近邻', AutoEncoder='多视角聚类')
#     MODEL_NAMES_REV = {}
#     MODEL_NAMES = {}
#     for key, val in _NAMES:
#         val2 = f'基于{val}的异常检测方法'
#         MODEL_NAMES[key] = val2
#         MODEL_NAMES[val2] = key


class MyWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pgbEvaluator.reset()
        self.ui.leNumTrain.setValidator(QIntValidator(1, 10000, self))
        self.ui.leNumTest.setValidator(QIntValidator(1, 5000, self))
        self.ui.leOutlierRate.setValidator(QDoubleValidator(0.1, 0.5, 2, self))
        self.ui.lbProgress.setText('就绪')
        self.ui.lbImage.setScaledContents(True)
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.ui.cbModelName.addItems(MODEL_ZOO.model_list)
        self.ui.cbModelName.setCurrentText('KNN')

        from NAME import NAME
        self.ui.centralwidget.setWindowTitle(NAME)

        self.ui.leNumTrain.setText(str(self.data_config.n_train))
        self.ui.leNumTest.setText(str(self.data_config.n_test))
        self.ui.leOutlierRate.setText(str(self.data_config.contamination))
        self.ui.leNumFeas.setText(str(self.data_config.n_features))
        self.ui.leSeed.setText(str(self.data_config.seed))
        self.job: RunEvaluator = None

    @pyqtSlot(str)
    def on_cbModelName_currentTextChanged(self, value: str):
        LOG.info(f'cbModelName: {value}')
        self.model_config.name = value

    @pyqtSlot()
    def on_leSeed_editingFinished(self):
        value = self.ui.leSeed.text()
        LOG.info(f'leSeed: {value}')
        try:
            self.data_config.seed = int(value)
        except ValueError:
            pass

    @pyqtSlot()
    def on_leNumTrain_editingFinished(self):
        value = self.ui.leNumTrain.text()
        LOG.info(f'leNumTrain: {value}')
        try:
            self.data_config.n_train = int(value)
        except ValueError:
            pass

    @pyqtSlot()
    def on_leNumTest_editingFinished(self):
        value = self.ui.leNumTest.text()
        LOG.info(f'leNumTest: {value}')
        try:
            self.data_config.n_test = int(value)
        except ValueError:
            pass

    @pyqtSlot()
    def on_leOutlierRate_editingFinished(self):
        value = self.ui.leOutlierRate.text()
        LOG.info(f'leOutlierRate: {value}')
        try:
            self.data_config.contamination = float(value)
        except ValueError:
            pass

    @pyqtSlot()
    def on_leNumFeas_editingFinished(self):
        value = self.ui.leNumFeas.text()
        LOG.info(f'leNumFeas: {value}')
        try:
            self.data_config.n_features = int(value)
        except ValueError:
            pass

    @pyqtSlot()
    def on_pbRunDetect_clicked(self):
        LOG.info(f'pbRunDetect clicked')
        if self.job is not None:
            QMessageBox.warning(self, '警告', '检测进行中，请等待',
                                QMessageBox.StandardButton.Yes,
                                QMessageBox.StandardButton.Yes)
            return

        pgb = self.ui.pgbEvaluator
        pgb.reset()
        pgb.setRange(0, len(RunEvaluator.ACTION_LIST) - 1)
        self.ui.lbProgress.setText('检测中')
        self.job = RunEvaluator(
            parent=self,
            data_config=self.data_config,
            model_config=self.model_config,
            slot_dict=self.build_slot_dict(),
        )
        self.job.start()

    ACTION_TO_PROGRESS = dict(
        load_data='数据加载完成',
        load_model='模型加载完成',
        detect='检测完成',
        visualize='可视化完成',
    )

    def reset_job(self):
        if self.job is None:
            return
        self.job.quit()
        self.job.wait()
        self.job.deleteLater()
        self.job = None

    def build_slot_dict(self):

        def on_visualize(tag: str, image: str):
            on_progress(tag)
            label = self.ui.lbImage
            image = QtGui.QPixmap(image).scaled(label.width(), label.height())
            assert not image.isNull()
            label.setPixmap(image)
            LOG.info(f'Label on {image}, tag {tag}')
            self.reset_job()

        def on_error(tag: str, msg: str):
            LOG.info(f'Error {msg}, tag {tag}')
            QMessageBox.warning(self, '错误', msg,
                                QMessageBox.StandardButton.Yes,
                                QMessageBox.StandardButton.Yes)
            self.reset_job()

        def on_progress(tag: str):
            assert tag in self.ACTION_TO_PROGRESS
            text = self.ACTION_TO_PROGRESS[tag]
            self.ui.lbProgress.setText(text)
            LOG.info(f'Progress {tag} => {text}')
            pgb = self.ui.pgbEvaluator
            val_pgb = RunEvaluator.ACTION_LIST.index(tag)
            assert val_pgb != -1
            pgb.setValue(val_pgb)

        slot_dict = {tag: on_progress for tag in RunEvaluator.ACTION_LIST}
        slot_dict.update(visualize=on_visualize, error=on_error)
        return slot_dict


class RunEvaluator(QThread):
    sig_load_data = pyqtSignal(str)
    sig_load_model = pyqtSignal(str)
    sig_detect = pyqtSignal(str)
    sig_visualize = pyqtSignal(str, str)
    sig_error = pyqtSignal(str, str)

    # error not in here!!
    ACTION_LIST = [
        'load_data',
        'load_model',
        'detect',
        'visualize',
    ]

    def __init__(
        self,
        parent,
        data_config: DataConfig,
        model_config: ModelConfig,
        slot_dict: Dict[str, Callable],
    ):
        super().__init__(parent)
        self.data_config = data_config
        self.model_config = model_config
        self.evaluator = DetectionEvaluator()
        for key, slot in slot_dict.items():
            try:
                sig = getattr(self, f'sig_{key}')
            except AttributeError:
                continue
            sig.connect(slot)

    def get_args(self, key):
        if key == 'load_data':
            return (self.data_config, )
        if key == 'load_model':
            return (self.model_config, )
        return tuple()

    def run(self) -> None:
        for key in self.ACTION_LIST:
            action = getattr(self.evaluator, key)
            args = self.get_args(key)
            try:
                ret = action(*args)
            except Exception as e:
                self.sig_error.emit('error', str(e))
                traceback.print_exc()
                LOG.warning(f'Error in action {key}')
                return
            else:
                sig = getattr(self, f'sig_{key}')
                if key == 'visualize':
                    sig.emit(key, str(ret))
                else:
                    sig.emit(key)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
