This is an implementation of [1] using Tensorflow. No attempt has been done in actually training the network. This does a forward and backward pass on a batch of images from the KITTI dataset.

# Usage
pip3 install pykitti
python3 main.py --frames_stop=50 --frame_step=5 --learning_rate=0.005 --corres=1000

# Data Loading
pykitti is used to load the images from both left and right cameras.
The volodyne points are projected into the images using pykitti. Note that the way pixel coordinates should be extracted from the values returned is not clear to me, I have for now just rounded to the nearest integer, this should be fixed before actually training the network. 

# Architecture
## Fully Convolutional Network
The first part of the network follows the implementation of GoogleNet up until the inception4a layer exept for the local response normalisation layers as suggested in [2].

## Convolutional Spatial Transformer
To build the convolutional spatial transformer we have a purely convolutional localisation network that returns the values of thetas to apply to each patch. We generate patches of given kernel size at each input. The transformation itself uses the implementation from David Dao. The patches are then merged back together before having a convolution applied with the given kernel size.

## Feature Extraction
We project the output feature back into the input space using bilinear interpolation. We then calculate the values at the given correspondence points.

## Hard Negative Mining with KNN
From the values of the feature from the first network at the corresondence points we get the nearest neighbours in the output feature of the second network. We use this to mine negatives if the nearest neighbour does not correspond to the correspondence point in the second image.

## Correspondence Contrastive Loss
Using both extracted feature at correspondence points for positive pairs and negative pairs as described above, we calculate the loss for all images.

## Not implemented
The accuracy metric using PCK was not implemented at this point.

# References
[1] Choy et. al. Universal Correspondence Network
[2] Choy et. al. Supplemental Materials for Universal Correspondence Network
