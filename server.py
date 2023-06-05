from flask import Flask, render_template, request, make_response
from flask_cors import CORS
from src.OutlierDetect import DetectionEvaluator, DataConfig, ModelConfig
import logging
import base64

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


@app.route("/run", methods=["GET"])
def run():
    params = parse_params()
    ev = DetectionEvaluator()
    cfg = DataConfig(
        n_train=params["training_set"],
        n_test=params["testing_set"],
        n_features=params["feature_dims"],
        seed=params["random_seed"],
        contamination=params["outlier_ratio"],
    )
    logging.info("Load data begins")
    ev.load_data(config=cfg)

    logging.info("Load model begins")
    cfg = ModelConfig(name=params["select_model"])
    ev.load_model(config=cfg)

    logging.info("Detect begins")
    ev.detect()

    logging.info("Visualize begins")
    image = ev.visualize()
    image = base64.b64encode(image.read_bytes()).decode()

    return make_response(dict(image=image))


if __name__ == "__main__":
    app.run(debug=True)
