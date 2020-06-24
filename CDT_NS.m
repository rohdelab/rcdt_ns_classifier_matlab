% 1D CDT Nearest Subspace Classifier
% adapted from Hasnat's 2D RCDT version
%
clc; clear; close all;

dataset = 'synthetic_1D/';
datadir = ['data/' dataset];
numClass = 2;

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
    im_train = cat(2,im_train,xxO);
    label_train = cat(2,label_train,label);
    
    % test set
    xxO = [];
    label = [];
    load([datadir 'testing/dataORG_' num2str(cls) '.mat']);
    im_test = cat(2,im_test,xxO);
    label_test = cat(2,label_test,label);
end
disp('Loaded data')

%% calculate CDT for both train and test sets

rm_edge = 1;
eps=1e-8;

% train set
disp('Calculate CDT of train signals')
if exist([datadir 'Xtrain.mat'])
    load([datadir 'Xtrain.mat'])
else
    Xtrain = [];
    for i = 1:size(im_train,2)
        I = im_train(:,i); % zero padding to both ends
        I = abs(I)+eps;
        I_domain = linspace(0,1,length(I));    % domain of 1D signal
        Ihat_domain = linspace(0,1,length(I)); % domain of CDT
        Ihat = CDT(I_domain, I/sum(I), Ihat_domain, rm_edge); % CDT of each sample
        Xtrain = cat(2, Xtrain, Ihat(:));
    end
    save([datadir 'Xtrain.mat'],'Xtrain','label_train')
end

% test set
disp('Calculate CDT of test data')
if exist([datadir 'Xtest.mat'])
    load([datadir 'Xtest.mat'])
else
    Xtest = [];
    for i = 1:size(im_test,2)
        I = im_test(:,i); % zero padding to both ends
        I = abs(I)+eps;
        I_domain = linspace(0,1,length(I));    % domain of 1D signal
        Ihat_domain = linspace(0,1,length(I)); % domain of CDT
        Ihat = CDT(I_domain, I/sum(I), Ihat_domain, rm_edge); % CDT of each sample
        Xtest = cat(2, Xtest, Ihat(:));
    end
    save([datadir 'Xtest.mat'],'Xtest','label_test')
end

%% FIT: Calculate the basis vectors for each class
trainSamples=[2,4,16,64,128,256,512];

for j=1:length(trainSamples)

    len_subspace = 0;
    for cls=0:numClass-1
        ind = find(label_train==cls);           % find train samples corresponding to class 'cls'
        ind_sub = randsample(ind,trainSamples(j)); % control the number of train samples to fit the model using 
                                                % 'trainSamples' variable; all the samples can also be used
                                                % by setting 'ind_sub = ind'
        classSamples = Xtrain(:,ind_sub);

        % calculate basis vectors using SVD
        [uu,su,vu]=svds(classSamples,size(classSamples,2));
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
            len_subspace = basis_ind(1);  % len_subspace is max over all classes
        end 
    end

    %% PREDICT: classify the test samples
    for cls=0:numClass-1
        B = basis(cls+1).V;
        B = B(:,1:len_subspace);
        Xproj = B*(B'*Xtest);               % projection of the test sample on the subspace
        Dproj = Xtest - Xproj;
        D(cls+1,:) = sqrt(sum(Dproj.^2,1));
    end
    [~,Ytest] = min(D);                     % predict the class label of the test sample
    Ytest = Ytest - 1;                      % class labels are defined from 0, but matlab index starts from 1

    Accuracy(j) = numel(find(Ytest==label_test))/length(Ytest);
end
Accuracy
%% PLOT the accuracy values
figure(1)
ph=semilogx(trainSamples,Accuracy,'r-o');
set(gca,'XTick',trainSamples,'XTickLabels',trainSamples,'FontSize',20,'LineWidth',2.0)
xlabel('Training Samples','FontSize',20)
ylabel('Classification Accuracy','FontSize',20)
set(ph,'LineWidth',2.0)
