clc; clear; close all;

dataset = 'MNIST/';
datadir = ['data/' dataset];
numClass = 10;
trainSamples = 1024;   % number of samples per class for training

%% load train and test image data
im_train = [];
label_train = [];
im_test = [];
label_test = [];
for cls=0:numClass-1
    % train set
    xxO = [];
    label = [];
    load([datadir 'training/dataORG_' num2str(cls) '.mat']);
    im_train = cat(3,im_train,xxO);
    label_train = cat(2,label_train,label);
    
    % test set
    xxO = [];
    label = [];
    load([datadir 'testing/dataORG_' num2str(cls) '.mat']);
    im_test = cat(3,im_test,xxO);
    label_test = cat(2,label_test,label);
end

%% calculate RCDT for both train and test sets
I_domain = [0, 1];
Ihat_domain = [0, 1];
theta_seq = 0:4:179;
rm_edge = 1;

% train set
disp('Calculate RCDT of train images')
if exist([datadir 'Xtrain.mat'])
    load([datadir 'Xtrain.mat'])
else
    Xtrain = [];
    for i = 1:size(im_train,3)
        I = squeeze(im_train(:,:,i));
        Ihat = RCDT(I_domain, I, Ihat_domain, theta_seq, rm_edge);
        Xtrain = cat(2, Xtrain, Ihat(:));
    end
    save([datadir 'Xtrain.mat'],'Xtrain','label_train')
end

% test set
disp('Calculate RCDT of test images')
if exist([datadir 'Xtest.mat'])
    load([datadir 'Xtest.mat'])
else
    Xtest = [];
    for i = 1:size(im_test,3)
        I = squeeze(im_test(:,:,i));
        Ihat = RCDT(I_domain, I, Ihat_domain, theta_seq, rm_edge);
        Xtest = cat(2, Xtest, Ihat(:));
    end
    save([datadir 'Xtest.mat'],'Xtest','label_test')
end


%% FIT: Calculate the basis vectors for each class
len_subspace = 0;
for cls=0:numClass-1
    ind = find(label_train==cls);
    ind_sub = randsample(ind,trainSamples);
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

%% PREDICT: classify the test samples
for cls=0:numClass-1
    B = basis(cls+1).V;
    B = B(:,1:len_subspace);
    Xproj = (B*B')*Xtest;
    Dproj = Xtest - Xproj;
    D(cls+1,:) = sqrt(sum(Dproj.^2,1));
end
[~,Ytest] = min(D);
Ytest = Ytest - 1;

Accuracy = numel(find(Ytest==label_test))/length(Ytest)




