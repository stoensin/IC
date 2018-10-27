### STEP 1


> Download the visual_genome dataset [visual_genome](http://visualgenome.org/)

> you should download the images part 1 (9.2 GB), part 2 (5.47 GB) with extract to directory 'VG/images';Download image meta data (17.62 MB) with extract to directory 'VG/1.2' ;Download region descriptions (712.07 MB) with extract to directory  'VG/1.2'  

> Data preparation : goto visual_genome folder for detail


### STEP 2


> download the faster r-cnn pretrained model from : [tf_faster_rcnn](https://github.com/endernewton/tf-faster-rcnn), and you should already 'make' the tf_faster_rcnn framework before start this step.

### STEP3

#### train
>  bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]


* dataset: 'visual_genome_1.2'
* net: 'res50' or 'res101'
* ckpt_to_init: pretrained model to be initialized with. get by step2
* data_dir: the data directory where you save the outputs after step 1.
* step: (int)
    - step 1: fix convnet weights
    - stpe 2: finetune convnets weights
    - step 3: add context fusion, but fix convnets weights
    - step 4: finetune the whole model.

#### test
> bash scripts/dense_cap_demo.sh [ckpt_path] [vocab_path]

* ckpt_path: the checkpoint path like './output/ckpt '
* vocab_path:. the vocabulary file where you put like:'/output/ckpt/vocabulary.txt'
* the pretrained model is here [model](https://drive.google.com/drive/folders/1AXlZREmP7fVi5qtHRWPyYwxiI4XpHle1?usp=sharing)

> After this command will write results into the folder vis/data. We have provided a web-based visualizer to view these results; to use it, change to the vis directory and start a local HTTP server:

```
cd vis
python -m http.server 8181
```
Then point your web browser to http://localhost:8181/view_results.html.



## References
* Dense captioning with joint inference and visual context [densecap](https://github.com/linjieyangsc/densecap)
* Adapted web-based visualizer from [jcjohnson](https://github.com/jcjohnson)'s [densecap repo](https://github.com/jcjohnson/densecap)
* The Faster-RCNN framework inherited from repo [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) by [endernewton](https://github.com/endernewton)
