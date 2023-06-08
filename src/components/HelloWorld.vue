<template>
  <div id="main-body" class="frame">
    <params-panel></params-panel>
    <image-canvas></image-canvas>
    <status-bar></status-bar>
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
