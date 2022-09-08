"""
这个文件要实现对异常检测框架PyOD的封装。
为Application提供接口。我们的输入：
1. 数据集，一般是numpy或者pandas格式；
2. 方法的种类（名称）；
3. 方法的超参数；
我们的输出：
1. 方法运行的结果；
2. 其他异常情况；
3. 可视化的结果；
"""
from pandas import Timestamp

import_start = Timestamp.now()
from sklearn.manifold import TSNE
from dataclasses import dataclass
import logging
from typing import Optional, Type
from joblib import Parallel, delayed, Memory
from numpy import ndarray
from pyod.models.base import BaseDetector
from pathlib import Path as P
from typing import Dict

VERBOSE = 999
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("[OutlierDetect]")
PROJ_DIR = P(__file__).parent
TMP_DIR = PROJ_DIR.joinpath("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJ_DIR.joinpath('.cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY = Memory(location=CACHE_DIR, verbose=VERBOSE)
NUM_JOBS = 4
import_dur = Timestamp.now() - import_start
LOG.info(f'Import: {import_dur.seconds}')


def make_directory_handy(dir: P):
    dir.mkdir(exist_ok=True, parents=True)


def tsne(X: ndarray) -> ndarray:
    return TSNE(n_jobs=2).fit_transform(X)


@dataclass
class DataConfig:
    contamination: float = 0.1
    n_train: int = 200
    n_test: int = 100
    n_features: int = 10
    seed: int = 32

    @property
    def contamination_percent(self):
        return int(100 * self.contamination)

    def load_data(self) -> 'Data':

        @MEMORY.cache
        def _load_data(self: DataConfig):
            from pyod.utils.data import generate_data

            d = Data()
            d.X_train, d.X_test, d.y_train, d.y_test = generate_data(
                n_train=self.n_train,
                n_test=self.n_test,
                contamination=self.contamination,
                n_features=self.n_features,
                random_state=self.seed,
            )
            d.X_train2d, d.X_test2d = Parallel(2)(
                delayed(tsne)(X) for X in [d.X_train, d.X_test])
            return d
    
        return _load_data(self)


class ModelZoo:
    """
    负责查找模型名字对应的模型类。
    并进行懒加载，因为导入pyod包非常耗时。
    """

    LOG = logging.getLogger("[ModelZoo]")

    RAW_TEXT = """\
    #     from pyod.models.abod import ABOD
    #     from pyod.models.auto_encoder import AutoEncoder
    #     from pyod.models.cblof import CBLOF
    #     from pyod.models.hbos import HBOS
    #     from pyod.models.iforest import IForest
    #     from pyod.models.knn import KNN
    #     from pyod.models.lof import LOF
    #     from pyod.models.mcd import MCD
    #     from pyod.models.ocsvm import OCSVM
    #     from pyod.models.pca import PCA\
    """

    @classmethod
    def discover_models(self) -> Dict[str, str]:
        """Load a list of available models.

        :return: a dict from class name to module name.
        :rtype: Dict[str, str]
        """
        import re

        PAT = re.compile(r"\s*\#\s*from pyod\.models\.(\w+) import (\w+)")
        data = self.RAW_TEXT.splitlines()
        data = list(map(lambda x: PAT.match(x).groups(), data))
        data = list(map(lambda x: tuple(reversed(x)), data))
        self.LOG.info(f"Valid models {data}")

        return dict(data)

    def __init__(self) -> None:
        self.module_map = self.discover_models()
        self.model_cache = {}

    def load(self, name: str) -> type:
        """Load a model class by its namae

        :param name: the class name of the model.
        :type name: str
        :return: the type object.
        :rtype: type
        """
        assert name in self.module_map, f"Bad model {name}"
        if name in self.model_cache:
            return self.model_cache[name]

        modname = self.module_map[name]  # Module name.
        ctx = {}
        exec(f"from pyod.models.{modname} import {name}", ctx)
        modelcls = ctx[name]
        self.model_cache[name] = modelcls
        return modelcls

    @property
    def model_list(self):
        """
        Return all valid names of models.
        """
        return list(self.module_map.keys())


MODEL_ZOO = ModelZoo()


@dataclass
class Data:
    """
    封装了一个有监督的数据集。该数据集是生成的。
    """

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    X_train2d = None
    X_test2d = None

    # 保存load传入的参数。
    config: dict = None

    @classmethod
    def load(self, contamination=0.1, n_train=200, n_test=100, n_features=100):
        """
        加载数据集。

        :param contamination: _description_, defaults to 0.1
        :type contamination: _type_, optional
        :param n_train: _description_, defaults to 200
        :type n_train: _type_, optional
        :param n_test: _description_, defaults to 100
        :type n_test: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        config = locals().copy()
        config.pop("self")
        from pyod.utils.data import generate_data

        d = Data()
        d.config = config
        d.X_train, d.X_test, d.y_train, d.y_test = generate_data(
            n_train=n_train,
            n_test=n_test,
            contamination=contamination,
            n_features=n_features,
        )
        from sklearn.manifold import TSNE
        d.X_train2d = TSNE().fit_transform(d.X_train)
        d.X_test2d = TSNE().fit_transform(d.X_test)

        return d

    def __repr__(self) -> str:
        return f"Data(config={self.config})"


@dataclass
class DetectionResult:
    """
    检测结果。基于它可以进行各种可视化和展示。
    """

    # 用到的模型的名字。
    clf_name: str = None
    # 用到的数据集。
    data: Data = None
    # get the prediction labels and outlier scores of the training data
    y_train_pred = None  # binary labels (0: inliers, 1: outliers)
    y_train_scores = None  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = None  # outlier labels (0 or 1)
    y_test_scores = None  # outlier scores
    y_test_pred_confidence = None

    def visualize(self) -> P:
        clf_name, data = self.clf_name, self.data
        from pyod.utils.example import visualize
        import os

        temp = P.cwd()
        image = TMP_DIR.joinpath(f"{clf_name}.png")
        try:
            image.unlink()
        except FileNotFoundError:
            pass

        os.chdir(TMP_DIR)
        visualize(
            clf_name,
            show_figure=False,
            save_figure=True,
            X_train=data.X_train2d,
            y_train=data.y_train,
            X_test=data.X_test2d,
            y_test=data.y_test,
            y_train_pred=self.y_train_pred,
            y_test_pred=self.y_test_pred,
        )
        os.chdir(temp)
        assert image.is_file()
        return image


@dataclass
class DetectionEvaluator:
    """
    主导一次检测的全过程，包括：
    1. 加载数据；
    2. 加载模型；
    3. 训练模型；
    4. 模型预测；
    5. 预测结果可视化；
    """

    config: DetectionConfig
    model: BaseDetector = None
    data: Data = None
    result: DetectionResult = None

    LOG = logging.getLogger("[Evaluator]")

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.LOG.info(f"init with config {config}")

    @property
    def clf_name(self):
        return self.config.model_name

    def load_model(self):
        config = self.config
        modelcls = MODEL_ZOO.load(config.model_name)

        self.model = modelcls(
            contamination=config.contamination,
            **(config.model_config or {}),
        )
        self.LOG.info(f"Model loaded: {self.model}")

    def load_data(self):
        config = self.config
        self.data = Data.load(
            contamination=config.contamination,
            n_train=config.n_train,
            n_test=config.n_test,
            n_features=config.n_features,
        )
        self.LOG.info(f"Data loaded: {self.data}")

    def fit_model(self):
        assert self.data is not None
        assert self.model is not None

        self.model.fit(self.data.X_train, self.data.y_train)
        self.LOG.info(f"Model fit")

    def predict(self) -> DetectionResult:
        result = DetectionResult(
            clf_name=self.config.model_name,
            data=self.data,
        )
        result.y_train_pred = self.model.labels_
        result.y_train_scores = self.model.decision_scores_
        result.y_test_pred, result.y_test_pred_confidence = self.model.predict(
            self.data.X_test, return_confidence=True)
        self.result = result
        self.LOG.info(f"Model predict")
        return result

    def visualize(self):
        image = self.result.visualize()
        self.LOG.info(f"Visualize {image}")
        return image


class RunEvaluator(QThread):
    LOG = logging.getLogger("[RunEvaluator]")

    sig_load_data = pyqtSignal(str)
    sig_load_model = pyqtSignal(str)
    sig_predict = pyqtSignal(str)
    sig_fit_model = pyqtSignal(str)
    sig_visualize = pyqtSignal(str, str)
    sig_error = pyqtSignal(str, str)

    # error not in here!!
    ACTION_LIST = [
        'load_data',
        'load_model',
        'fit_model',
        'predict',
        'visualize',
    ]

    def __init__(
        self,
        parent,
        config: DetectionConfig,
        slot_dict: Dict[str, Callable],
    ):
        super().__init__(parent)
        self.evaluator = DetectionEvaluator(config)
        for key, slot in slot_dict.items():
            try:
                sig = getattr(self, f'sig_{key}')
            except AttributeError:
                continue
            sig.connect(slot)
            self.LOG.info(f'Connect sig-slot {key}')

    def run(self) -> None:
        for key in self.ACTION_LIST:
            action = getattr(self.evaluator, key)
            try:
                ret = action()
            except Exception as e:
                self.sig_error.emit('error', str(e))
                traceback.print_exc()
                self.LOG.warning(f'Error in action {key}')
                return
            else:
                sig = getattr(self, f'sig_{key}')
                if key == 'visualize':
                    sig.emit(key, str(ret))
                else:
                    sig.emit(key)


if __name__ == "__main__":
    pass
