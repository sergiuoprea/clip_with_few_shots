<div align="center">    
 
# Few-Zero shot learning using CLIP architecture as backbone
## Proudly implemented in Pytorch Lightning

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>
 
## Description   
This is a simple 3-layer multi-class classifier working on top of features extracted from [CLIP architecture](https://github.com/openai/CLIP). The provided dataset is imbalanced, so we have used a WeightedRandomSampler to ensure the DataLoader retrieves the samples with equal probability. Also, training set was splitted into the train and validation splits using the latter to choice the best performing model. On the other side, we have tested ViT-B/32 and ViT-B/16 CLIP models. The latter provided the best results in our case.

## Goals
* Design a system to recognize novel object types from a few images used for training.
* Start from pretrained [CLIP architecture](https://github.com/openai/CLIP)
* Few shot task: develop a classification model trained and tested on the provided data.
* Zero shot task: as an extension of the few shot task, and using no data for training.

## Provided dataset
* For training: a dataset (/data/train) consisting of a small set of training images, i.e. 10-20 samples per object class.
* For testing: a dataset (/data/test) providing some images reserved to evaluate the system.

## How to run (Docker)
Run build.sh script. This will automatically install all the dependencies in requirements.txt and clone the project in the docker container (/src/clip_with_few_shots).
```bash
# build docker container  
./build.sh

# run docker container
./run.sh
 ```   
 Next, navigate to /src/clip_with_few_shots and run main.py. This will, train, validate and test the model properly.
 ```bash
# module folder
cd /src/clip_with_few_shots

# run module
python main.py    
```

## Some results 

### Confusion matrix
The horizontal and vertical axes indices correspond with the following classes:
```
['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'truck']
```
<img src="plots/normalized_conf_matrix.png" width="800">

We notice a confussion between cars and trucks due to the similarities between both.

### Training and validation accuracies

<img src="plots/logs_train_acc_per_epoch.png" width="800">
<img src="plots/logs_valid_acc_per_epoch.png" width="800">
