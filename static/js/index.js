const PARAMS = [
    'select_model',
    'training_set',
    'testing_set',
    'outlier_ratio',
    'feature_dims',
    'random_seed',
];

function Parameters() {
    this.select_model = document.getElementById("select-model").value;
    this.training_set = document.getElementById("training-set").value;
    this.testing_set = document.getElementById("testing-set").value;
    this.outlier_ratio = document.getElementById("outlier-ratio").value;
    this.feature_dims = document.getElementById("feature-dims").value;
    this.random_seed = document.getElementById("random-seed").value;

    console.log('Load parameters');
    for (let key of PARAMS) {
        console.log(`${key} => ${this[key]}`);
    }
    this.as_url = function () {
        var out = [];
        for (let key of PARAMS) {
            out.push(`${key}=${this[key]}`);
        }
        return out.join('&');
    }
}

var parameters;

function updateParams() {
    parameters = new Parameters();
}

window.onload = () => { updateParams(); };

const URL = 'http://127.0.0.1:5000';

function run() {
    updateParams();
    const param_str = parameters.as_url();
    fetch(`${URL}/run?${param_str}`)
        .then((data) => data.json())
        .then((data) => {
            var imgdata = data['image'];
            imgdata = `data:image/png;base64,${imgdata}`;
            const canvas = document.getElementById("image-canvas");
            canvas.innerHTML = `<img src=${imgdata} alt="Detection Result"/>`;
            console.log(`Get run ${data}`);
        });
}
