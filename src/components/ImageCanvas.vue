<template>
    <div id="image-canvas" class="frame">
        <img :src="result_image ? result_image : default_image" alt="Detection Result" />
    </div>
</template>

<script>
import '@/assets/common.css'

export default {
    name: 'ImageCanvas',
    props: ['result_image'],
    data() {
        return {
            default_image: require('@/assets/splash.png'),
        }
    },
}
</script>

<style scoped>
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
</style>