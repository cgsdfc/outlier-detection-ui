<template>
    <div id="parameters" class="frame">
        <div class="param">
            <label for="select-model">Select a Model</label>
            <select v-model="select_model" id="select-model">
                <option v-for="m in methods" :key="m" :value="m" :label="m"></option>
            </select>
        </div>
        <div class="param">
            <label for="training-set">#Training Set</label>
            <input type="number" id="training-set" v-model="training_set">
        </div>

        <div class="param">
            <label for="testing-set">#Testing Set</label>
            <input type="number" id="testing-set" v-model="testing_set">
        </div>

        <div class="param">
            <label for="outlier-ratio">Outliers%</label>
            <input type="number" id="outlier-ratio" v-model="outlier_ratio" step=".01" min="0" max="1">
        </div>
        <div class="param">
            <label for="feature-dims">Feature Dims</label>
            <input type="number" id="feature-dims" v-model="feature_dims">
        </div>
        <div class="param">
            <label for="random-seed">Random Seed</label>
            <input type="number" id="random-seed" v-model="random_seed">
        </div>
    </div>
</template>

<script>
import '@/assets/common.css'

export const PARAMS = [
    'select_model',
    'training_set',
    'testing_set',
    'outlier_ratio',
    'feature_dims',
    'random_seed',
];

export default {
    name: 'ParamsPanel',
    data() {
        return {
            select_model: 'KNN',
            training_set: 1000,
            testing_set: 100,
            outlier_ratio: 0.1,
            feature_dims: 10,
            random_seed: 42,
            methods: [
                "ABOD",
                "HBOS",
                "IForest",
                "KNN",
                "LOF",
                "MCD",
                "OCSVM",
                "PCA",
            ]
        }
    },
    computed: {
        get_params: function () {
            console.log('Load parameters');
            var out = [];
            for (let key of PARAMS) {
                console.log(`${key} => ${this[key]}`);
                out.push(`${key}=${this[key]}`);
            }
            return out.join('&');
        },
    }
}
</script>

<style scoped>
.param {
    display: flex;
    flex-direction: column;
    text-align: left;
    text-transform: uppercase;
    font-weight: bold;
    justify-content: center;
}

/* Grid items */
#parameters {
    display: flex;
    /* flex-wrap: wrap; */
    justify-content: space-evenly;
    align-items: center;
    margin-bottom: 5px;
    background-color: lightblue;
}

#select-model,
#training-set,
#testing-set,
#outlier-ratio,
#feature-dims,
#random-seed {
    font-size: large;
    width: 100%;
    max-width: 120px;
    margin-right: 5px;
    margin-bottom: 5px;
    padding: 5px;
    border-radius: 3px;
    border: 1px solid #ccc;
}

label {
    width: 100%;
    max-width: 150px;
    color: black;
}

/* Form styles */
select,
input[type="number"] {
    display: inline-block;
}
</style>