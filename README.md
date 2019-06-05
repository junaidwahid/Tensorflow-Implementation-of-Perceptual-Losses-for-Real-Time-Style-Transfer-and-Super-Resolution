# Tensorflow-Implementation-of-Perceptual-Losses-for-Real-Time-Style-Transfer-and-Super-Resolution
Hi buddies. This is my first github repository. I coded this 2 years back, but due to time unavailability I could not able to upload it.
The way code is written is might looks like old tensorflow style but all things are present in this repository.

This repository contains the implementation of  Justin Johnson's Paper "Perceptual Losses for Real-Time Style Transfer  and Super-Resolution" in Tensorflow

The paper is using an algorithm which takes content from content image and style from given style image and generates combination of both.Here is an example:
<img src="https://i.ibb.co/H2G17Mq/example.png" alt="example" border="0">

<b>Setup</b><br>
Dataset:
<ul><li><a href="http://cocodataset.org/#home">Coco Dataset</a></li></ul>

Dependecies:
<ul>
  <li>Tensorflow</li>
  <li>Cuda</li>
  <li>Hdf5</li>
  <li>Opencv</li>
  <li>scikit-learn</li>
  <li>Other Basic Libraries</li>
</ul>

After installing all these dependecies, then you need to download the pretrained weigths of squeezenet. The reason behind sequeezent is that in paper they are extracting features from it and it is also one of the lighest pretrained model.

<a href="https://github.com/avoroshilov/tf-squeezenet" style="margin-left:20px">                 Download Squeezenet </a>

<b>Usage</b><br>

<b>Training</b><br>
Basic usage:
```python train.py -param <"init" or "restore"> -num_epoch <int> -model_path <./model.ckpt> -train_size <int> -batch_size <int> -style_img <./style_image.jpg>  -dataset_path <./dataset_git.hdf5> -squeezenet_path <./squeezenet.ckpt>```

Detail of parameters:
```
-param: use "init" (when you want to train it from scratch) or use "restore" (when you 
        want to use checkpointed weigths)
-num_epoch: The number of iteration of dataset
-model_path: The path where you want to save your final model or restore weights from 
             checkpoints
-train_size: The number of images in dataset
-batch_size : Batch size
-style_img: The path of style image
-dataset_path: The path where HDF5 dataset file is saved.
-squeezenet_path: The squeezenet model weights path



```
<b>Testing</b><br>
Basic Usage:
```
python test.py -content_img <./content_image.jpg> -style_img <./style_img.jpg> -output_path <./pic2.jpg> -squeezenet_path  <./squeezenet.ckpt>
```
Detail of parameters:
```
-content_img: The path of content image
-style_img: The path of style image
-output_path: The path where you want to save the output image
-squeezenet_path: The squeezenet model weights path
```





             
       




    
