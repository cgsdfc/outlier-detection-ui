const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/api': {
        target: process.env.VUE_APP_API_URL, // 将请求转发到的目标服务器
        changeOrigin: true, // 开启跨域
        pathRewrite: {
          '^/api': '' // 将 URL 中的 /api 去掉
        }
      }
    }
  }
})
