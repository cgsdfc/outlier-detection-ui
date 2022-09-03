# 20220901 导师项目：异常值检测可视化

纪要：
1. 可视化形式要多样，多种图形，3D；
2. 后台的代码找现成的Python代码，确保有几个基于聚类的方法；
3. 前端基于pyqt；

https://github.com/yzhao062/pyod

https://pyod.readthedocs.io/en/latest/example.html

生成仿真数据

https://pyod.readthedocs.io/en/latest/pyod.utils.html#pyod.utils.data.generate_data

所有模型

https://pyod.readthedocs.io/en/latest/pyod.models.html

模型列表
```
-*- coding: utf-8 -*-
from .abod import ABOD
from .auto_encoder import AutoEncoder
from .cblof import CBLOF
from .combination import aom, moa, average, maximization
from .feature_bagging import FeatureBagging
from .hbos import HBOS
from .iforest import IForest
from .knn import KNN
from .lof import LOF
from .mcd import MCD
from .ocsvm import OCSVM
from .pca import PCA

__all__ = ['ABOD',
           'AutoEncoder',
           'CBLOF',
           'aom', 'moa', 'average', 'maximization',
           'FeatureBagging',
           'HBOS',
           'IForest',
           'KNN',
           'LOF',
           'MCD',
           'OCSVM',
           'PCA']

```
