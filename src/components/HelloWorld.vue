<template>
  <div id="main-body" class="frame">

    <div id="parameters" class="frame">
      <div class="param">
        <label for="select-model">Select a Model</label>
        <select v-model="state.select_model" id="select-model">
          <option v-for="m in METHODS" :key="m" :value="m" :label="m"></option>
        </select>
      </div>
      <div class="param">
        <label for="training-set">#Training Set</label>
        <input type="number" id="training-set" v-model="state.training_set">
      </div>

      <div class="param">
        <label for="testing-set">#Testing Set</label>
        <input type="number" id="testing-set" v-model="state.testing_set">
      </div>

      <div class="param">
        <label for="outlier-ratio">Outliers%</label>
        <input type="number" id="outlier-ratio" v-model="state.outlier_ratio" step=".01" min="0" max="1">
      </div>
      <div class="param">
        <label for="feature-dims">Feature Dims</label>
        <input type="number" id="feature-dims" v-model="state.feature_dims">
      </div>
      <div class="param">
        <label for="random-seed">Random Seed</label>
        <input type="number" id="random-seed" v-model="state.random_seed">
      </div>
    </div>

    <div id="image-canvas" class="frame">
      <!-- Add JavaScript to dynamically generate the image and display it in this canvas -->
      <img :src="display_image" alt="Detection Result" />
    </div>

    <div id="status-bar" class="frame">
      <span id="status-label" class="frame">{{ state.status }}</span>
      <progress :value="state.pgb" max="100" id="progress-bar"></progress>
      <button id="run-btn" class="frame" @click="run()">RUN</button>
    </div>

  </div>
</template>

<script setup>
import splash from '@/assets/splash.png'
import { computed } from '@vue/reactivity';
import { reactive } from '@vue/reactivity';

const PARAMS = [
  'select_model',
  'training_set',
  'testing_set',
  'outlier_ratio',
  'feature_dims',
  'random_seed',
];

const URL = '/api';

const METHODS = [
  "ABOD",
  "HBOS",
  "IForest",
  "KNN",
  "LOF",
  "MCD",
  "OCSVM",
  "PCA",
]

const state = reactive({
  select_model: 'KNN',
  training_set: 1000,
  testing_set: 100,
  outlier_ratio: 0.1,
  feature_dims: 10,
  random_seed: 42,
  status: 'Ready',
  pgb: 0,
  result_image: null,
  default_image: splash,
})

const display_image = computed(() => state.result_image ?
  state.result_image : state.default_image)

function get_params() {
  console.log('Load parameters');
  var out = [];
  for (let key of PARAMS) {
    console.log(`${key} => ${state[key]}`);
    out.push(`${key}=${state[key]}`);
  }
  return out.join('&');
}

function run() {
  const param_str = get_params();
  console.log(`params_str ${param_str}`)
  state.pgb = 0;
  state.status = 'Ready';

  fetch(`${URL}/load_data?${param_str}`)
    .then(() => {
      state.pgb = 25;
      state.status = 'Data Loaded';
      return fetch(`${URL}/load_model?${param_str}`);
    })
    .then(() => {
      state.pgb = 50;
      state.status = 'Model Loaded';
      return fetch(`${URL}/detect?${param_str}`);
    })
    .then(() => {
      state.pgb = 75;
      state.status = 'Detected';
      return fetch(`${URL}/visualize?${param_str}`);
    })
    .then((data) => data.json())
    .then((data) => {
      var imgdata = data['image'];
      imgdata = `data:image/png;base64,${imgdata}`;
      state.result_image = imgdata;
      state.pgb = 100;
      state.status = 'Done';
    });
}

</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
/* Global styles */
* {
  box-sizing: border-box;
}

body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
}

.frame {
  margin: 5px;
  padding: 10px;
  background-color: lightgray;
  border-radius: 10px;
  box-shadow: rgba(0, 0, 0, 0.24) 0px 3px 8px;
}

#main-body {
  background-color: lightgray;
  display: flex;
  flex-direction: column;
}

#heading {
  text-align: center;
  font-size: large;
  background-color: white;
}

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

#image-canvas {
  overflow: hidden;
  min-height: 500px;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: white;
}

div img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

#status-bar {
  display: flex;
  align-items: center;
  justify-content: space-evenly;
  border: 1px solid #ccc;
  padding: 10px;
  background-color: lightblue;
}

#status-label {
  font-weight: bold;
  margin-right: 10px;
  max-width: 200px;
  width: 100px;
  border-radius: 3px;
  text-align: center;
}

progress {
  width: 500px;
  height: 20px;
  border-radius: 10px;
  background-color: #ddd;
}

progress::-webkit-progress-value {
  background-color: #4caf50;
  border-radius: 10px;
}

progress::-moz-progress-bar {
  background-color: #4caf50;
  border-radius: 10px;
}

/* Form styles */
select,
input[type="number"] {
  display: inline-block;
}

#run-btn {
  background-color: #007bff;
  color: white;
  padding: 10px 20px;
  border-radius: 3px;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s ease-in-out;
  max-width: 100px;
  width: 300px;
  height: 40px;
}
</style>
