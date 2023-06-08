<template>
  <div id="main-body" class="frame">

    <div id="parameters" class="frame">
      <div class="param">
        <label for="select-model">Select a Model</label>
        <select name="select-model" id="select-model">
          <option value="ABOD">ABOD</option>
          <option value="HBOS">HBOS</option>
          <option value="IForest">IForest</option>
          <option value="KNN">KNN</option>
          <option value="LOF">LOF</option>
          <option value="MCD">MCD</option>
          <option value="OCSVM">OCSVM</option>
          <option value="PCA">PCA</option>
          <!-- Add more options as needed -->
        </select>
      </div>
      <div class="param">
        <label for="training-set">#Training Set</label>
        <input type="number" id="training-set" name="training-set" value="1000">
      </div>

      <div class="param">
        <label for="testing-set">#Testing Set</label>
        <input type="number" id="testing-set" name="testing-set" value="100">
      </div>

      <div class="param">
        <label for="outlier-ratio">Outliers%</label>
        <input type="number" id="outlier-ratio" name="outlier-ratio" step=".01" min="0" max="1" value=".1">
      </div>
      <div class="param">
        <label for="feature-dims">Feature Dims</label>
        <input type="number" id="feature-dims" name="feature-dims" value="10">
      </div>
      <div class="param">
        <label for="random-seed">Random Seed</label>
        <input type="number" id="random-seed" name="random-seed" value="42">
      </div>
    </div>

    <div id="image-canvas" class="frame">
      <!-- Add JavaScript to dynamically generate the image and display it in this canvas -->
      <img src="../assets/splash.png" alt="Detection Result" />
    </div>

    <div id="status-bar" class="frame">
      <span id="status-label" class="frame">Ready</span>
      <progress value="0" max="100" id="progress-bar"></progress>
      <button id="run-btn" class="frame" onclick="run()">RUN</button>
    </div>

  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
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
