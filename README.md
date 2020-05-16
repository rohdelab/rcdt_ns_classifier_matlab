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

### Calculate RCDT

1. Define some basic parameters of RCDT:

```matlab
I_domain = [0, 1];
Ihat_domain = [0, 1];
theta_seq = 0:4:179;
rm_edge = 1;
```
2. Calculate RCDT for the train images:

```matlab
Xtrain = [];
for i = 1:size(im_train,3)
    I = squeeze(im_train(:,:,i));
    Ihat = RCDT(I_domain, I, Ihat_domain, theta_seq, rm_edge);
    Xtrain = cat(2, Xtrain, Ihat(:));
end
```

3. Calculate RCDT for the test images:

```matlab
Xtest = [];
for i = 1:size(im_test,3)
    I = squeeze(im_test(:,:,i));
    Ihat = RCDT(I_domain, I, Ihat_domain, theta_seq, rm_edge);
    Xtest = cat(2, Xtest, Ihat(:));
end
```

### Calculate the basis vectors for each class

```matlab
len_subspace = 0;
for cls=0:numClass-1
    ind = find(label_train==cls);           % find train samples corresponding to class 'cls'
    ind_sub = randsample(ind,trainSamples); % control the number of train samples to fit the model using 
                                            % 'trainSamples' variable; all the samples can also be used
                                            % by setting 'ind_sub = ind'
    classSamples = Xtrain(:,ind_sub);
    
    % calculate basis vectors using SVD
    [uu,su,vu]=svd(classSamples);
    s=diag(su);
    eps= 1e-4;
    indx=find(s>eps);
    V=uu(:,indx);
    
    basis(cls+1).V = V;
    
    % take basis components with atleast 99% variance
    S = cumsum(s);
    S = S/max(S);
    basis_ind = find(S>=0.99);
    if len_subspace < basis_ind(1)
        len_subspace = basis_ind(1);
    end 
end
```

### Test the model

```matlab
%% PREDICT: classify the test samples
for cls=0:numClass-1
    B = basis(cls+1).V;
    B = B(:,1:len_subspace);
    Xproj = (B*B')*Xtest;               % projection of the test sample on the subspace
    Dproj = Xtest - Xproj;
    D(cls+1,:) = sqrt(sum(Dproj.^2,1)); % distance between test sample and its projection
end
[~,Ytest] = min(D);                     % predict the class label of the test sample
Ytest = Ytest - 1;                      % class labels are defined from 0, but matlab index starts from 1

Accuracy = numel(find(Ytest==label_test))/length(Ytest)
```

#### The above steps have also been compiled in a single matlab script ```RCDT_NS.m``` which runs the RCDT-NS classifier on MNIST dataset.

# Publication for Citation
Please cite the following publication when publishing findings that benefit from the codes provided here.

#### Shifat-E-Rabbi M, Yin X, Rubaiyat AHM, Li S, Kolouri S, Aldroubi A, Nichols JM, Rohde GK. "Radon cumulative distribution transform subspace modeling for image classification." arXiv preprint arXiv:2004.03669 (2020). [[Paper](https://arxiv.org/abs/2004.03669)]
