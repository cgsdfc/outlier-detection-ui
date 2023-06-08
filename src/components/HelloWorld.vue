<template>
  <div id="main-body" class="frame">
    <!-- 传出params-str -->
    <params-panel @params-changed="(e) => { params_str = e }"></params-panel>
    <!-- 传入result-image -->
    <image-canvas :result_image="result_image"></image-canvas>
    <!-- 传入params-str，传出result-image -->
    <status-bar :params_str="params_str" @result-ready="(e) => { result_image = e }"></status-bar>
  </div>
</template>

<script>
import '@/assets/common.css';
import ImageCanvas from './ImageCanvas.vue';
import ParamsPanel from './ParamsPanel.vue';
import StatusBar from './StatusBar.vue';

export default {
  name: 'HelloWorld',
  components: {
    'params-panel': ParamsPanel,
    'image-canvas': ImageCanvas,
    'status-bar': StatusBar,
  },
  data() {
    return {
      result_image: null,
      params_str: '',
    }
  },
}

</script>

<style scoped>
/* Global styles */
* {
  box-sizing: border-box;
}

#main-body {
  background-color: lightgray;
  display: flex;
  flex-direction: column;
}
</style>
