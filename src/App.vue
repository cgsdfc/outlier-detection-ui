<template>
  <OutlierDetectDemo />
</template>

<script>
import OutlierDetectDemo from './components/OutlierDetectDemo.vue'

export default {
  name: 'App',
  components: {
    OutlierDetectDemo: OutlierDetectDemo
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
}
</style>
