<template>
  <div id="app">
    <TopNav></TopNav>
    <transition name="fade">
        <router-view class="app-view"></router-view>
    </transition>
    <lg-preview></lg-preview>
    <BottomNav></BottomNav>
  </div>
</template>

<script>
import TopNav from './components/common/TopNav.vue'
import BottomNav from './components/common/BottomNav.vue'
export default {
  name: 'app',
  created: function(){
  },
  components: {
    TopNav,
    BottomNav,
  }
}
</script>

<style>
  html,body{
    background-color: #424a53;
  }

  ol, ul {
      margin: 0;
      padding: 0;
      list-style: none;
  }

  p{
    -webkit-margin-before: 0px;
    -webkit-margin-after: 0px;
  }

  *, :after, :before {
      box-sizing: border-box;
  }

  #app {
    width: 100%;
    height: 100%;
    background-color: #39434f;
  }

  .app-view{
    width: 100%;
    height: 100%;
    overflow: auto;
    position: absolute;
    -webkit-overflow-scrolling: touch;
  }

  .fade-enter-active, .fade-leave-active {
    transition: opacity .5s
  }
  .fade-enter, .fade-leave-active {
    opacity: 0
  }
  .lg-preview-title{
    font-size: 1rem!important;
    height: 4rem!important;
    line-height: 14px!important;
    letter-spacing:1px;
  }
  .mint-spinner-snake{
    margin: 0 auto;
  }
  @media screen and (min-width:760px){
    #app {
      width: 600px;
      margin: 0 auto;
    }
    .app-view{
      width: 600px;
    }
    .lg-preview-wrapper{
        width: 600px!important;
    }
  }
</style>
