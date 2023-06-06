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
    var pgb = document.getElementById('progress-bar');
    var status = document.getElementById("status-label");
    status.innerHTML = 'Ready';
    pgb.value = 0;
    // document.getElementById("image-canvas").innerHTML = '<img src="./static/img/splash.png" alt="Detection Result"/>';

    fetch(`${URL}/load_data?${param_str}`)
        .then(() => {
            pgb.value = 25;
            status.innerHTML = 'Data Loaded';
            return fetch(`${URL}/load_model?${param_str}`);
        })
        .then(() => {
            pgb.value = 50;
            status.innerHTML = 'Model Loaded';
            return fetch(`${URL}/detect?${param_str}`);
        })
        .then(() => {
            pgb.value = 75;
            status.innerHTML = 'Detected';
            return fetch(`${URL}/visualize?${param_str}`);
        })
        .then((data) => data.json())
        .then((data) => {
            var imgdata = data['image'];
            imgdata = `data:image/png;base64,${imgdata}`;
            const canvas = document.getElementById("image-canvas");
            canvas.innerHTML = `<img src=${imgdata} alt="Detection Result"/>`;
            pgb.value = 100;
            status.innerHTML = 'Done';
        });
}
