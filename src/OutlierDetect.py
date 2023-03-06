# MIT License
# 
# Copyright (c) 2022 Cong Feng
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from sklearn.manifold import TSNE
from dataclasses import dataclass
import logging
from typing import Type
from joblib import Parallel, delayed, Memory
from numpy import ndarray
from pyod.models.base import BaseDetector
from pathlib import Path as P
import os


def ensure_dir(dir: P):
    dir.mkdir(exist_ok=True, parents=True)
    return dir


from typing import Dict

VERBOSE = 999
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("[OutlierDetect]")
PROJ_DIR = P(__file__).parent
TMP_DIR = ensure_dir(PROJ_DIR.joinpath("tmp"))
CACHE_DIR = ensure_dir(PROJ_DIR.joinpath('.cache'))
MEMORY = Memory(location=CACHE_DIR, verbose=VERBOSE)
NUM_JOBS = 1

EAGER = os.getenv('ODQT_EAGER', '')
if EAGER:
    LOG.info('EAGER loading pyod modules...')
    import src.PYODLIBS as PYODLIBS
    LOG.info('Done')
else:
    PYODLIBS = None


def tsne(X: ndarray) -> ndarray:
    return TSNE(n_jobs=1).fit_transform(X)


@dataclass
class DataConfig:
    contamination: float = 0.1
    n_train: int = 500
    n_test: int = 300
    n_features: int = 50
    seed: int = 32

    @property
    def contamination_percent(self):
        return int(100 * self.contamination)

    def load_data(self) -> 'Data':

        @MEMORY.cache
        def _load_data(self: DataConfig):
            from pyod.utils.data import generate_data

            d = Data()
            d.config = self
            d.X_train, d.X_test, d.y_train, d.y_test = generate_data(
                n_train=self.n_train,
                n_test=self.n_test,
                contamination=self.contamination,
                n_features=self.n_features,
                random_state=self.seed,
            )
            LOG.info(f'Begin TSNE')
            d.X_train2d, d.X_test2d = Parallel(1)(
                delayed(tsne)(X) for X in [d.X_train, d.X_test])
            LOG.info('End TSNE')

            return d

        return _load_data(self)


@dataclass
class ModelConfig:
    name: str = 'KNN'

    def load_model(self, contamination: float):

        @MEMORY.cache
        def _load_model(self: ModelConfig, contamination: float):
            cls = MODEL_ZOO.load(self.name)
            ins = cls(contamination=contamination)
            return Model(model_config=self, model=ins)

        return _load_model(self, contamination)


@dataclass
class Model:
    model_config: ModelConfig
    model: BaseDetector

    @property
    def name(self):
        return self.model_config.name

    def detect(self, data: 'Data'):

        @MEMORY.cache
        def _detect(self: Model, data: Data):
            self.model.fit(data.X_train, data.y_train)
            res = DetectionResult()
            res.y_train_pred = self.model.labels_
            res.y_train_scores = self.model.decision_scores_
            res.y_test_pred, res.y_test_pred_confidence = self.model.predict(
                data.X_test, return_confidence=True)
            return res

        return _detect(self, data)


class ModelZoo:
    """
    负责查找模型名字对应的模型类。
    并进行懒加载，因为导入pyod包非常耗时。
    """

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

        @MEMORY.cache
        def _discover_models(raw_text: str):
            import re

            PAT = re.compile(r"\s*\#\s*from pyod\.models\.(\w+) import (\w+)")
            data = raw_text.splitlines()
            data = list(map(lambda x: PAT.match(x).groups(), data))
            data = list(map(lambda x: tuple(reversed(x)), data))

            return dict(data)

        return _discover_models(self.RAW_TEXT)

    def __init__(self, **renames) -> None:
        self.module_map = self.discover_models()
        self.real2display = renames
        self.display2real = {val: key for key, val in renames.items()}

    def load(self, disname: str) -> Type[BaseDetector]:
        """Load a model class by its namae

        :param name: the class name of the model.
        :type name: str
        :return: the type object.
        :rtype: type
        """
        name = self.dis2real(disname)
        assert name in self.module_map, f"Bad model {name}"
        if EAGER:
            return getattr(PYODLIBS, name)

        @MEMORY.cache
        def _load_modelcls(self: ModelZoo, name: str):
            modname = self.module_map[name]  # Module name.
            ctx = {}
            exec(f"from pyod.models.{modname} import {name}", ctx)
            modelcls = ctx[name]
            return modelcls

        return _load_modelcls(self, name)

    @property
    def model_list(self):
        """
        Return all valid names of models.
        """
        disnames = list(map(self.real2dis, self.module_map.keys()))
        return disnames

    def real2dis(self, real: str):
        return self.real2display.get(real, real)

    def dis2real(self, dis: str):
        return self.display2real.get(dis, dis)


MODEL_ZOO = ModelZoo(AutoEncoder='MVC')


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
    config: DataConfig = None

    def __repr__(self) -> str:
        return f"Data(config={self.config})"


@dataclass
class DetectionResult:
    """
    检测结果。基于它可以进行各种可视化和展示。
    """
    # get the prediction labels and outlier scores of the training data
    y_train_pred = None  # binary labels (0: inliers, 1: outliers)
    y_train_scores = None  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = None  # outlier labels (0 or 1)
    y_test_scores = None  # outlier scores
    y_test_pred_confidence = None


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

    model: Model = None
    data: Data = None
    result: DetectionResult = None

    @property
    def data_config(self):
        return self.data.config

    @property
    def model_config(self):
        return self.model.model_config

    def load_model(self, config: ModelConfig):
        self.model = config.load_model(self.data_config.contamination)
        return self

    def load_data(self, config: DataConfig):
        self.data = config.load_data()
        return self

    def detect(self):
        self.result = self.model.detect(self.data)
        return self

    def visualize(self, parent: P = None):
        parent = parent or TMP_DIR
        parent = ensure_dir(parent.absolute())

        import os
        temp = P.cwd()
        os.chdir(parent)
        clf_name, data, res = self.model.name, self.data, self.result
        from pyod.utils.example import visualize
        visualize(
            clf_name=clf_name,
            show_figure=False,
            save_figure=True,
            X_train=data.X_train2d,
            y_train=data.y_train,
            X_test=data.X_test2d,
            y_test=data.y_test,
            y_train_pred=res.y_train_pred,
            y_test_pred=res.y_test_pred,
        )
        os.chdir(temp)
        image = parent.joinpath(f'{clf_name}.png')
        return image


if __name__ == "__main__":
    ev = DetectionEvaluator()
    ev.load_data(DataConfig()).load_model(
        ModelConfig()).detect().visualize(TMP_DIR)
