# MATLAB implementation of Radon Cumulative Distribution Transform Nearest Subspace (RCDT-NS) classifier
This repository contains the MATLAB implementation of the Radon cumulative distribution transform nearest subspace (RCDT-NS) classifier proposed in the paper titled "Radon cumulative distribution transform subspace models for image classification". A python implementation of the classifier using PyTransKit (Python Transport Based Signal Processing Toolkit) package is also available in: https://github.com/rohdelab/rcdt_ns_classifier.

## Walk-through of the RCDT-NS classifier implementation
### Organize datasets

Organize an image classification dataset as follows:

1. Download the image dataset, and seperate it into the `training` and `testing` sets.
2. For the `training` set: 
    - Save images from different classes into separate `.mat` files. Dimension of the each `.mat` file would be `M x N x K`, where `M x N` is the size of the images and `K` is the number of samples per class.
    - Name of the mat file would be `dataORG_<class_index>.mat`. For example, `dataORG_0.mat` and `dataORG_1.mat` would be two mat files for a binary class problem.
    - Save the mat files in the `./data/training` directory.
3. For the `testing` set:
    - The first two steps here are the same as the first two steps for the `training` set.
    - Save the mat files in the `./data/testing` directory.

### Read Data

Load the train and test images from the .mat files provided in the dataset directory.

```matlab
im_train = []; label_train = []; 
im_test = []; label_test = [];
for cls=0:numClass-1
    % train set
    xxO = []; label = [];
    load([datadir 'training/dataORG_' num2str(cls) '.mat']);
    im_train = cat(3,im_train,xxO);
    label_train = cat(2,label_train,label);
    
    % test set
    xxO = []; label = [];
    load([datadir 'testing/dataORG_' num2str(cls) '.mat']);
    im_test = cat(3,im_test,xxO);
    label_test = cat(2,label_test,label);
end
```

# Publication for Citation
Please cite the following publication when publishing findings that benefit from the codes provided here.

#### Shifat-E-Rabbi M, Yin X, Rubaiyat AHM, Li S, Kolouri S, Aldroubi A, Nichols JM, Rohde GK. "Radon cumulative distribution transform subspace modeling for image classification." arXiv preprint arXiv:2004.03669 (2020). [[Paper](https://arxiv.org/abs/2004.03669)]
