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

from dataclasses import dataclass
from distutils.command.config import config
from pprint import pprint
import logging
from typing import Any
logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    """
    配置一次运行的所有参数。
    """
    model_name: str
    contamination: float
    n_train: int
    n_test: int
    model_config: dict


class ModelZoo:
    LOG=logging.getLogger('[Model]')

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

        PAT=re.compile(r'\s*\#\s*from pyod\.models\.(\w+) import (\w+)')
        data=self.RAW_TEXT.splitlines()
        data=list(map(lambda x: PAT.match(x).groups(), data))
        self.LOG.info(f'valid models {data}')

        return dict(data)

    def __init__(self) -> None:
        self.model_map = self.discover_models()
        self.model_cache = {}

    def load(self, name: str) -> type:
        """
        Load a model class by its lower-case name.
        """
        assert name in self.model_map, f'Bad model {name}'
        if name in self.model_cache:
            return self.model_cache[name]

        clsname = self.model_map[name]
        ctx = {}
        exec(f'from pyod.models.{name} import {clsname}', ctx)
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

    X_train=None
    X_test=None
    y_train=None
    y_test=None
    config=None
    
    @classmethod    
    def load(self,contamination=0.1,n_train=200,n_test=100):
        config = locals().copy()
        config.pop('self')
        from pyod.utils.data import generate_data
        d = Data()
        d.config = config
        d.X_train, d.X_test, d.y_train, d.y_test = generate_data(
            n_train=n_train, n_test=n_test, contamination=contamination)
        return d

    def __repr__(self) -> str:
        return f'Data(config={self.config})'


@dataclass
class Evaluator:
    config: Config
    model: 'BaseDetector'
    data: Data

    LOG=logging.getLogger('[Evaluator]')

    def __init__(self, config: Config) -> None:

        self.config = config
        modelcls = MODEL_ZOO.load(config.model_name)

        from pyod.models.base import BaseDetector

        self.model: BaseDetector = modelcls(contamination=config.contamination,
            **config.model_config)
        self.LOG.info(f'Model instance created: {self.model}')
            
        self.data = Data.load(
            contamination=config.contamination,
            n_train=config.n_train,
            n_test=config.n_test,
        )
        # self.LOG.info(f'Data loaded: {')
        

if __name__=='__main__':
    d=Data.load()
    print(d)
    