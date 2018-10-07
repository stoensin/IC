<template>

  <div class="wallpaper_box">

    <img-inputer class="inputer" auto-upload action='api/upload' v-model="file"  :onSuccess="upsuccess" theme="light" size="small"/>

    <ul class="wallpaper_list"  v-infinite-scroll="loadMore" infinite-scroll-disabled="loaded" infinite-scroll-distance="255">
      <li v-for="item in list">
        <img v-lazy="item.pic" v-preview="item.pic" :src="item.pic" :alt="item.captions[0].caption">
        <Card v-bind:picInfo="item"><span slot="tips_text" class="tips_text">{{ item.captions[0].caption }}</span></Card>
      </li>
    </ul>

    <p class="wallpaper_list_loading" v-show="isLoadShow"><mt-spinner type="snake"></mt-spinner></p>
    <p class="wallpaper_list_loaded" v-show="loaded">NO MORE</p>
  </div>
</template>

<script>
import Card from '../common/Card.vue'
export default {
  name: 'Index',
  data () {
    return {
      file: null,
      list : [],
      page : 1,
      isLoadShow : true,
      loading : false,
      loaded : false,
    }
  },
  methods : {
    loadMore() {
      this.login()
      if (!this.loading){
        this.loading = true;
        setTimeout(() => {
          this.$http.get("api/").then((response) => {
            if (response.data.data == false){
              this.loaded = true;
              this.isLoadShow = false;
            }else{
              let length = response.data.data.length;
              for (let i = 0; i < length; i++) {
                if (localStorage.love){
                  let data = JSON.parse(localStorage.love);
                  response.data.data[i].show = false;
                  for(let j=0; j< data.length; j++) {
                    if (data[j].id == response.data.data[i].id){
                      response.data.data[i].show = true;
                      break;
                    }
                  }
                }else{
                  localStorage.love = "[]";
                  response.data.data[i].show = false;
                }

                this.list.push(response.data.data[i]);
              }
              this.page++;
            }
            this.loading = false;
          },(err) => {
            console.log('Load Pic Data Failed',err)
          })
        }, 500)
      }
    },
    login() {
        this.$http.post("api/login").then((response) => {
          if (response.data.data == true){

          }
        },(err) => {
          console.log('App login failed',err)
        })
    },
    upsuccess(res,file){
      let temp=[]
      temp= res
      temp['id']= this.list.length
      this.add(temp)
    },
    add(item){
      this.list.unshift(item)
    },
    rev(list){
      return list.reverse()
    }
  },
  components: {
    Card,
  },
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  .wallpaper_box{
    background-color: #39434f;
    padding-top: 45px;
    padding-bottom: 50px;
    padding-left: 8px;
    padding-right: 8px;
  }

  .wallpaper_list{
    margin: 0;
    padding: 0;
    text-align: center;
    list-style-type: none;
  }
  .wallpaper_list li{
    width: 100%;
    height: auto;
    margin: 0 auto;
    margin-top: 8px;
  }

  .wallpaper_list li img{
    display: block;
    width: 100%;
    height: auto;
    border-top-right-radius: 5px;
    border-top-left-radius: 5px;
    -moz-border-top-right-radius: 5px;
    -moz-border-top-left-radius: 5px;
    -webkit-border-top-right-radius: 5px;
    -webkit-border-top-left-radius: 5px;
  }

  .wallpaper_list li img[lazy=loading] {
    width: 100%;
    height: 222px;
    margin: auto;
    background: url("../../assets/img/loading.png");
    background-size: cover;
  }

  .wallpaper_list_loading {
    text-align: center;
    height: 50px;
    line-height: 50px;
    padding-top: 12px;
  }

  .wallpaper_list_loaded {
    text-align: center;
    height: 50px;
    line-height: 50px;
    font-size: 12px;
    color: #999;
  }
  .tips {
    font-size: 12px;
  }
  .tips .tips_text{
    width: 84%;
    overflow: hidden;
    height: 40px;
    text-align: left;
  }
  .wallpaper_list_loading div {
    display: inline-block;
    vertical-align: middle;
    margin-right: 5px;
  }
  .inputer{
    display: block;
    margin: 0 auto;
    margin-top: 6px;
  }
</style>
