# Deep Learning applications on PyTorch

This repository contains some Deep Learning personal applications, using different methods (CNN, RNN), to solve diverse problems.

## Table Of Contents

### Convolutional Neural Networks (CNN)

* **Image classification**: 
    * In a [first notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/CNN/Image_classification_transfer.ipynb), we use **transfer learning** from a pre-trained Residual Neuronal Network (**ResNet**) to classify images from the STL10 dataset.
We obtain a **94.9%** prediction accuracy (**over the 10 classes**).
    * In a [second notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/CNN/Image_classification_implemented.ipynb), we **implement** and train a **ResNet** from scratch, given that the previous pre-trained ResNet is not adapted to the small images of the CIFAR10 datset we want to classify. We obtain a **84.5%** prediction accuracy (**over the 10 classes**).


* **Image segmentation**: To be done... (U-Net)

* **Face recognition**: To be done... (FaceNet)


### Recurrent Neural Networks (RNN)

* **Embedding**: To be done...

* **Attention networks**: To be done...

* **LSTM**: To be done... (biblio)
