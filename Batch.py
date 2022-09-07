"""
批量运行数据生成和模型训练
"""

from pathlib import Path as P
from OutlierDetect import MODEL_ZOO
from joblib import Parallel, delayed

