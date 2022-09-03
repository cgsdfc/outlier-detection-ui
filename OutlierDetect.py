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
# Don't import large package here!
# Import when you need it.
from dataclasses import dataclass
from pprint import pprint
import logging
from typing import Any
from pyod.models.base import BaseDetector
from pathlib import Path as P

logging.basicConfig(level=logging.INFO)

PROJ_DIR = P(__file__).parent
TMP_DIR = PROJ_DIR.joinpath("tmp")
KEY_MODEL_NAME = "model_name"


@dataclass
class Config:
    """
    配置一次运行的所有参数。
    """

    # 模型的名字，比如 ABOD
    model_name: str
    # outlier 所占的比例
    contamination: float
    # 训练样本的数量
    n_train: int
    # 测试样本的数量
    n_test: int
    # 传给模型构造器的额外参数
    model_config: dict = None


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
    def discover_models(self):
        import re

        PAT = re.compile(r"\s*\#\s*from pyod\.models\.(\w+) import (\w+)")
        data = self.RAW_TEXT.splitlines()
        data = list(map(lambda x: PAT.match(x).groups(), data))
        self.LOG.info(f"Valid models {data}")

        return dict(data)

    def __init__(self) -> None:
        self.model_map = self.discover_models()
        self.model_cache = {}

    def load(self, name: str) -> type:
        """
        Load a model class by its lower-case name.
        """
        assert name in self.model_map, f"Bad model {name}"
        if name in self.model_cache:
            return self.model_cache[name]

        clsname = self.model_map[name]
        ctx = {}
        exec(f"from pyod.models.{name} import {clsname}", ctx)
        modelcls = ctx[clsname]
        self.model_cache[name] = modelcls
        return modelcls

    @property
    def model_list(self):
        """
        Return all valid names of models.
        """
        return list(self.model_map.keys())


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
    # 保存load传入的参数。
    config: dict = None

    @classmethod
    def load(self, contamination=0.1, n_train=200, n_test=100):
        config = locals().copy()
        config.pop("self")
        from pyod.utils.data import generate_data

        d = Data()
        d.config = config
        d.X_train, d.X_test, d.y_train, d.y_test = generate_data(
            n_train=n_train, n_test=n_test, contamination=contamination
        )
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
        from pyod.utils.example import visualize as visualize
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
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
            y_train_pred=self.y_train_pred,
            y_test_pred=self.y_test_pred,
        )
        os.chdir(temp)
        assert image.is_file()
        return image


@dataclass
class Evaluator:
    """
    主导一次检测的全过程，包括：
    1. 加载数据；
    2. 加载模型；
    3. 训练模型；
    4. 模型预测；
    5. 预测结果可视化；
    """

    config: Config
    model: BaseDetector = None
    data: Data = None
    result: DetectionResult = None

    LOG = logging.getLogger("[Evaluator]")

    def __init__(self, config: Config) -> None:
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
            self.data.X_test, return_confidence=True
        )
        self.result = result
        self.LOG.info(f"Model predict")
        return result

    def visualize(self):
        image = self.result.visualize()
        self.LOG.info(f"Visualize {image}")
        return image


if __name__ == "__main__":
    # P("xxxxxxxxxxxxx").unlink()
    evaluator = Evaluator(
        Config(
            model_name="knn",
            contamination=0.1,
            n_train=200,
            n_test=100,
        )
    )
    evaluator.load_data()
    evaluator.load_model()
    evaluator.fit_model()
    evaluator.predict()
    evaluator.visualize()

