// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import axios from 'axios'
import store from './store/index'

import './assets/css/normalize.css'
import 'mint-ui/lib/style.css'

import { Lazyload } from 'mint-ui';
import { InfiniteScroll } from 'mint-ui';
import { Spinner } from 'mint-ui';
import Icon from 'vue-svg-icon/Icon.vue';
import vuePicturePreview from 'vue-picture-preview'
import ImgInputer from 'vue-img-inputer'
import 'vue-img-inputer/dist/index.css'

Vue.use(vuePicturePreview);
Vue.use(Lazyload);
Vue.use(InfiniteScroll);
Vue.component(Spinner.name, Spinner);
Vue.component('icon', Icon);
Vue.component('ImgInputer', ImgInputer)

axios.defaults.withCredentials = true

Vue.prototype.$http = axios
Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  store,
  template: '<App/>',
  components: { App }
})
