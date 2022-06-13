# Deep Learning applications on PyTorch

This repository contains some Deep Learning personal applications, using different methods (CNN, RNN), to solve diverse problems.

## Table Of Contents

### Convolutional Neural Networks (CNN)

* **Image classification**: 
    * In a [first notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/CNN/Image_classification_transfer.ipynb), we use **transfer learning** from a pre-trained Residual Neuronal Network (**ResNet**) to classify images from the **STL10** dataset.
We obtain a **94.9%** prediction accuracy (**over the 10 classes**).
    * In a [second notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/CNN/Image_classification_implemented.ipynb), we **implement** and train a **ResNet** from scratch, given that the previous pre-trained ResNet is not adapted to the small images of the **CIFAR10** datset we want to classify. We obtain a **84.5%** prediction accuracy (**over the 10 classes**).

*Here are some sample images from the STL10 (left) and CIFAR10 (right) datasets:*
<img src="images/cifar10_stl10.png" width=100%>


* **Image segmentation**: Coming soon. (U-Net)

* **Face recognition**: Coming soon. (FaceNet, DeepFace)


### Recurrent Neural Networks (RNN)

* **Sentiment analysis**: We implement a sentiment analysis to associate stars (from 1 to 5) to the reviews of the **YelpReviewFull** dataset. In this analysis, **we use only 20% of the reviews** to train and test the model: this amounts already to 140,000 reviews! ...


* **Attention networks**: Coming soon.

* **Transformer networks**: Coming soon. 




## Some bibliography

To be completed: 2 books, jupyter notebooks, courses...
