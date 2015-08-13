# Google Street View Recognition

*finished on 28th Mar, 2015*

**I will add a convolutional neural network solution soon.**

[The competition](https://www.kaggle.com/c/street-view-getting-started-with-julia) focuses the task of identifying characters from Google Street View images. It differs from traditional character recognition because the data set contains different character fonts and the background is not the same for all images. 

This is actually my first data science competition. I was a consultant at that time and was on a business trip. I learned and coded in the hotel room after work, at nights and during weekends. A very cool experience:)

My final model was kNN with cosine similarity and matrix of HOG, which got 0.68914 ranking 7/36 at my last submission.

## Models

The models I tried are:

* kNN
* Random Forest
* Extra Trees

## Features

I tried these features from raw image:

* 20\*20 greyscale intensity
* 30\*30 greyscale intensity
* ...
* HOG (histogram of gradient)
* 0-1 standardization

I tried these distance metrics:

* corrcoef similarity
* cosine similarity
* Euclidean Distance

## Best Scores

* Extra Trees, 0-1-stand: 0.59936
* Random Forest, 0-1 stand: 0.52280
* kNN, Euc, 0-1 stand: 0.52305
* kNN, corr: 0.56121
* kNN, Euc, HOG(40\*40, 5\*5): 0.62162
* kNN, corr, HOG(40\*40, 5\*5): 0.65596
* kNN, cos, HOG(40\*40, 5\*5): 0.67727



