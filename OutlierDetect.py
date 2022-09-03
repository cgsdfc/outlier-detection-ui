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

from pprint import pprint


class Model:

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
        return dict(data)


if __name__=='__main__':
    # myglobal={}
    # exec('from pyod.models.auto_encoder import AutoEncoder', myglobal)
    # pprint(myglobal)
    # # pprint(vars(Model))



    pprint(Model.discover_models())
