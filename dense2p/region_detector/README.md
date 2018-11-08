# Faster R-CNN / Mask R-CNN on COCO
This example provides a minimal (2k lines) and faithful implementation of the following papers:

+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
+ [Mask R-CNN](https://arxiv.org/abs/1703.06870)
+ [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

with the support of:
+ Multi-GPU / distributed training
+ [Cross-GPU BatchNorm](https://arxiv.org/abs/1711.07240)
+ [Group Normalization](https://arxiv.org/abs/1803.08494)

## Dependencies
+ Python 3; OpenCV.
+ TensorFlow >= 1.6 (1.4 or 1.5 can run but may crash due to a TF bug);
+ pycocotools: `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/FasterRCNN/)
  from tensorpack model zoo.
+ [COCO data](http://cocodataset.org/#download). It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train201?.json
    instances_val201?.json
  train201?/
    COCO_train201?_*.jpg
  val201?/
    COCO_val201?_*.jpg
```

You can use either the 2014 version or the 2017 version of the dataset.
To use the common "trainval35k + minival" split for the 2014 dataset, just
download the annotation files `instances_minival2014.json`,
`instances_valminusminival2014.json` from
[here](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md)
to `annotations/` as well.

<sup>Note that train2017==trainval35k==train2014+val2014-minival2014, and val2017==minival2014</sup>


## Usage
### Train:

On a single machine:
```
./train.py --config \
    MODE_MASK=True MODE_FPN=True \
    DATA.BASEDIR=/path/to/COCO/DIR \
    BACKBONE.WEIGHTS=/path/to/ImageNet-R50-AlignPadding.npz \
```

To run distributed training, set `TRAINER=horovod` and refer to [HorovodTrainer docs](http://tensorpack.readthedocs.io/modules/train.html#tensorpack.train.HorovodTrainer).

Options can be changed by either the command line or the `config.py` file.
Recommended configurations are listed in the table below.

The code is only valid for training with 1, 2, 4 or >=8 GPUs.
Not training with 8 GPUs may result in different performance from the table below.

### Inference:

To predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model --config SAME-AS-TRAINING
```

To evaluate the performance of a model on COCO:
```
./train.py --evaluate output.json --load /path/to/COCO-R50C4-MaskRCNN-Standard.npz \
    --config SAME-AS-TRAINING
```

Several trained models can be downloaded in the table below. Evaluation and
prediction will need to be run with the corresponding training configs.

## Results

These models are trained on trainval35k and evaluated on minival2014 using mAP@IoU=0.50:0.95.
All models are fine-tuned from ImageNet pre-trained R50/R101 models in the [model zoo](http://models.tensorpack.com/FasterRCNN/).
Performance in [Detectron](https://github.com/facebookresearch/Detectron/) can be roughly reproduced.
Mask R-CNN results contain both box and mask mAP.

 

## Notes

[NOTES.md](NOTES.md) has some notes about implementation details & speed.
