<template>
    <div id="status-bar" class="frame">
        <span id="status-label" class="frame">{{ status }}</span>
        <!-- <progress :value="pgb" max="100" id="progress-bar"></progress> -->
        <el-progress :percentage="pgb" color="success" striped></el-progress>
        <!-- <button id="run-btn" class="frame" @click="run()">RUN</button> -->
        <el-button id="run-btn" type="primary" @click="run()">RUN</el-button>
    </div>
</template>

<script>

export const URL = '/api';

export default {
    name: 'StatusBar',
    props: ['params_str'],
    data() {
        return {
            pgb: 0,
            status: 'Ready',
            result_image: '',
        }
    },
    methods: {
        run() {
            const param_str = this.params_str;
            this.pgb = 0;
            this.status = 'Ready';

            fetch(`${URL}/load_data?${param_str}`)
                .then(() => {
                    this.pgb = 25;
                    this.status = 'Data Loaded';
                    return fetch(`${URL}/load_model?${param_str}`);
                })
                .then(() => {
                    this.pgb = 50;
                    this.status = 'Model Loaded';
                    return fetch(`${URL}/detect?${param_str}`);
                })
                .then(() => {
                    this.pgb = 75;
                    this.status = 'Detected';
                    return fetch(`${URL}/visualize?${param_str}`);
                })
                .then((data) => data.json())
                .then((data) => {
                    var imgdata = data['image'];
                    imgdata = `data:image/png;base64,${imgdata}`;
                    this.result_image = imgdata;
                    this.pgb = 100;
                    this.status = 'Done';
                    this.$emit("result-ready", this.result_image);
                });
        }
    }
}
</script>

<style scoped>
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
/*
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
} */

#run-btn {
    /* background-color: #007bff;
    color: white;
    padding: 10px 20px;
    border-radius: 3px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease-in-out; */
    max-width: 100px;
    width: 300px;
    height: 40px;
}
</style>