from flask import Flask, render_template, request, make_response
from flask_cors import CORS
from src.OutlierDetect import DetectionEvaluator, DataConfig, ModelConfig
import logging
import base64
import time

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

PARAMS = dict(
    select_model=str,
    training_set=int,
    testing_set=int,
    outlier_ratio=float,
    feature_dims=int,
    random_seed=int,
)

logging.info(PARAMS)
ev = DetectionEvaluator()


def parse_params():
    out = {}
    for name, type in PARAMS.items():
        val = request.args[name]
        val = type(val)
        out[name] = val
    return out


@app.route("/")
def index():
    return render_template("./index.html")


@app.route("/load_data", methods=["GET"])
def load_data():
    params = parse_params()
    cfg = DataConfig(
        n_train=params["training_set"],
        n_test=params["testing_set"],
        n_features=params["feature_dims"],
        seed=params["random_seed"],
        contamination=params["outlier_ratio"],
    )
    logging.info("Load data begins")
    ev.load_data(config=cfg)
    time.sleep(2)
    return dict(status='Data Loaded', step=1)


@app.route("/load_model", methods=["GET"])
def load_model():
    params = parse_params()
    logging.info("Load model begins")
    cfg = ModelConfig(name=params["select_model"])
    ev.load_model(config=cfg)
    time.sleep(2)
    return dict(status='Model Loaded', step=2)


@app.route("/detect", methods=["GET"])
def detect():
    logging.info("Detect begins")
    ev.detect()
    time.sleep(2)
    return dict(status='Detected', step=3)


@app.route("/visualize", methods=["GET"])
def visualize():
    logging.info("Visualize begins")
    image = ev.visualize()
    image = base64.b64encode(image.read_bytes()).decode()

    time.sleep(2)
    return dict(status='Done', step=4, image=image)


if __name__ == "__main__":
    app.run(debug=True)
