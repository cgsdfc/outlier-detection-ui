<template>
  <div id="main-body" class="frame">

    <div id="parameters" class="frame">
      <el-form inline v-model="state" label-position="top">
        <el-form-item label="Select a Model">
          <el-select v-model="state.select_model">
            <el-option v-for="m in METHODS" :label="m" :value="m"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="#Training Set">
          <el-input-number v-model="state.training_set"></el-input-number>
        </el-form-item>
        <el-form-item label="#Testing Set">
          <el-input-number v-model="state.testing_set"></el-input-number>
        </el-form-item>
        <el-form-item label="Outliers%">
          <el-input-number :precision=2 v-model="state.outlier_ratio"></el-input-number>
        </el-form-item>
        <el-form-item label="#Features">
          <el-input-number v-model="state.feature_dims"></el-input-number>
        </el-form-item>
        <el-form-item label="Random Seed">
          <el-input-number v-model="state.random_seed"></el-input-number>
        </el-form-item>
      </el-form>
    </div>

    <div id="image-canvas" class="frame">
      <!-- Add JavaScript to dynamically generate the image and display it in this canvas -->
      <img :src="result_image ? result_image : splash" alt="Detection Result" />
    </div>

    <div id="status-bar">
      <span id="status-label" class="frame">{{ status }}</span>
      <el-progress :percentage="pgb" style="width:80%" :stroke-width=30 :striped=true :striped-flow=true :show-text=false
        :color="green" :stroke-linecap="round"></el-progress>
      <el-button type="primary" :loading="0 < pgb && pgb < 100" @click="run()">RUN</el-button>
    </div>

  </div>
</template>

<script setup>
import splash from '@/assets/splash.png'
import { computed } from '@vue/reactivity';
import { reactive } from '@vue/reactivity';
import { ref } from 'vue';

const PARAMS = [
  'select_model',
  'training_set',
  'testing_set',
  'outlier_ratio',
  'feature_dims',
  'random_seed',
];

const URL = 'http://127.0.0.1:5000';

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
})

const status = ref('Ready')
const pgb = ref(0)
const result_image = ref('')

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
  pgb.value = 0;
  status.value = 'Ready';
  // result_image.value = '';
  console.log(`RUN!!!`)

  fetch(`${URL}/load_data?${param_str}`)
    .then(() => {
      pgb.value = 25;
      status.value = 'Data Loaded';
      console.log(`${status.value}`)
      return fetch(`${URL}/load_model?${param_str}`);
    })
    .then(() => {
      pgb.value = 50;
      status.value = 'Model Loaded';
      console.log(`${status.value}`)
      return fetch(`${URL}/detect?${param_str}`);
    })
    .then(() => {
      pgb.value = 75;
      status.value = 'Detected';
      console.log(`${status.value}`)
      return fetch(`${URL}/visualize?${param_str}`);
    })
    .then((data) => data.json())
    .then((data) => {
      var imgdata = data['image'];
      imgdata = `data:image/png;base64,${imgdata}`;
      result_image.value = imgdata;
      pgb.value = 100;
      status.value = 'Done';
      console.log(`${status.value}`)
      console.log(`${result_image ? 1 : 0}`)
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

/* progress {
  width: 500px;
  height: 20px;
  border-radius: 10px;
  background-color: #ddd;
} */


/* #run-btn {
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
} */
</style>
