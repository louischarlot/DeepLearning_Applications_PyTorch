# Deep Learning applications on PyTorch

This repository contains some personal applications of Deep Learning, using different methods (CNN, RNN), to solve various problems.

## Table Of Contents

### Convolutional Neural Networks (CNN)

* **Image classification**: 
    * In a [first notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/CNN/Image_classification_transfer.ipynb), we use **transfer learning** from a pre-trained Residual Neuronal Network (**ResNet**) to classify images from the **STL10** dataset.
We obtain a prediction accuracy of **94.9%** (**over the 10 classes**).
    * In a [second notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/CNN/Image_classification_implemented.ipynb), we **implement** and train a **ResNet** from scratch, given that the previous pre-trained ResNet is not suitable for the small images of the **CIFAR10** dataset we want to classify. We obtain a prediction accuracy of **84.5%** (**over the 10 classes**).

*Here are some sample images from the STL10 (left) and CIFAR10 (right) datasets:*
<img src="images/cifar10_stl10.png" width=100%>


* **Image segmentation**: Work in progress. (U-Net)

* **Face recognition**: Work in progress. (FaceNet, DeepFace)


### Recurrent Neural Networks (RNN)

* **Sentiment analysis**: In [this notebook](https://github.com/louischarlot/DeepLearning_Applications_PyTorch/blob/main/RNN/Sentiment_Analysis.ipynb), we implement a **sentiment analysis** to predict whether the reviews in the [YelpReviewFull dataset](https://pytorch.org/text/stable/datasets.html#yelpreviewfull) are **good**, **neutral**, or **bad**. Our RNN, consisting of 2 layers ([Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM)), gives a  **test accuracy of 80%**.
<br> An additional test,  **on two personal reviews** (one **positive** and one **negative**) which we wrote ourselves, yields the compelling results in the following Figure:
<img src="images/my_reviews.png" width=100%>


* **Attention networks**: Work in progress.

* **Transformer networks**: Work in progress. 




## Some recommended bibliography

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep learning](https://mitpress.mit.edu/books/deep-learning). MIT press.

- The [CS230 Deep Learning class](https://cs230.stanford.edu/) by Andrew Ng and Kian Katanforoosh.

- [Udacity](https://github.com/udacity/deep-learning-v2-pytorch) very interesting examples on PyTorch.

- [PyTorch official website](https://pytorch.org/), that contains a lot of useful explanations about PyTorch functions.




