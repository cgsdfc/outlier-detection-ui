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
批量运行数据生成和模型训练
"""

from pathlib import Path as P
from OutlierDetect import *

OUTPUT_DIR = ensure_dir(PROJ_DIR.joinpath('output'))


def make_data_config(contamination):
    return DataConfig(
        contamination=contamination,
        n_train=500,
        n_test=1000,
        n_features=500,
    )


def run(data_config: DataConfig, model_config: ModelConfig, root: P):
    c = data_config
    prefix = '-'.join(map(str, [c.n_train, c.n_test, c.n_features]))
    root = root.joinpath(prefix).joinpath(f'{c.contamination_percent}')
    DetectionEvaluator().load_data(c).load_model(
        model_config).detect().visualize(root)


def get_args(rate_list, model_list, root):
    from itertools import product

    for rate, name in product(rate_list, model_list):
        mc = ModelConfig(name)
        dc = make_data_config(rate)
        yield dc, mc, root


def batch_run():
    model_list = MODEL_ZOO.model_list
    rate_list = [0.1, 0.3, 0.5]

    Parallel(NUM_JOBS, verbose=VERBOSE)(
        delayed(run)(*args)
        for args in get_args(rate_list, model_list, OUTPUT_DIR))


if __name__ == "__main__":
    batch_run()
