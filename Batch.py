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
