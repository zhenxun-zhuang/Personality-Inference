function audio_reg(varargin)

% -------------------------------------------------------------------------
%prepare the data
% -------------------------------------------------------------------------

load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\train\features\fea_audio_comb_mfcc_train.mat');
fea_audio_comb_mfcc_train=reshape(fea_audio_comb_mfcc_train,[1,1,size(fea_audio_comb_mfcc_train,1),size(fea_audio_comb_mfcc_train,2)]);
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\train\labels_audio_train.mat') ;
labels_audio_train=labels_audio_train';
labels_audio_train=reshape(labels_audio_train,[1,1,size(labels_audio_train,1),size(labels_audio_train,2)]);
dim = size(fea_audio_comb_mfcc_train, 3);
num_train=size(fea_audio_comb_mfcc_train,4);

load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\val\features\fea_audio_comb_mfcc_val.mat');
fea_audio_comb_mfcc_val=reshape(fea_audio_comb_mfcc_val,[1,1,size(fea_audio_comb_mfcc_val,1),size(fea_audio_comb_mfcc_val,2)]);
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\val\labels_audio_val.mat') ;
labels_audio_val=labels_audio_val';
labels_audio_val=reshape(labels_audio_val,[1,1,size(labels_audio_val,1),size(labels_audio_val,2)]);
num_val=size(fea_audio_comb_mfcc_val,4);

imdb.meta.sets={'train','val'};
imdb.images.data=single(cat(4, fea_audio_comb_mfcc_train, fea_audio_comb_mfcc_val));
imdb.images.label=single(cat(4, labels_audio_train, labels_audio_val));
imdb.images.set = int8(ones(1, num_train+num_val));%reset
%imdb.images.set=[ones(1,num_train), ones(1,num_val)+1];

% -------------------------------------------------------------------------
% initialize a CNN architecture
% -------------------------------------------------------------------------

run('C:\Study\Programming\Matlab\MatConvNet\matconvnet-1.0-beta23\matlab\vl_setupnn.m');


% -------------------------------------------------------------------------
% 5-fold cross validation
% -------------------------------------------------------------------------


if 0
    cv_idx = ml_kFoldCV_Idxs(size(imdb.images.set,2), 5);
    for cv_ite=1:1
        imdb.images.set(cv_idx{cv_ite})=2;%mark as validation

        net = audio_reg_init(dim) ;
        % Add a loss (using a custom layer)
        net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;
        
        trainOpts.batchSize = 100 ;
        trainOpts.numEpochs = 300 ;
        trainOpts.continue = true ;
        trainOpts.gpus = [];
        trainOpts.learningRate = 0.001 ;
        trainOpts.expDir = sprintf('exp_cv\\audio\\%d',cv_ite) ;
        trainOpts.errorFunction = 'none' ;
        trainOpts = vl_argparse(trainOpts, varargin);

        % Take the average image out
        imageMean = mean(imdb.images.data(:,:,:,imdb.images.set==1),4) ;
        imdb.images.data = imdb.images.data - imageMean ;

        % Call training function in MatConvNet
        [~,~] = cnn_train(net, imdb, @getBatch, trainOpts) ;

        imdb.images.set(cv_idx{cv_ite})=1;%reset
    end
end


net = audio_reg_init_3layers(dim) ;

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% -------------------------------------------------------------------------
% train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 8001;
trainOpts.continue = true ;
trainOpts.gpus = [];
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'exp\audio_comb_mfcc' ;
trainOpts.errorFunction = 'none' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imageMean = mean(imdb.images.data(:,:,:,imdb.images.set==1),4) ;
imdb.images.data = imdb.images.data - imageMean ;

% Call training function in MatConvNet
[net,~] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
if trainOpts.gpus==1
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.meta.imageMean = imageMean ;
save('exp\audio_comb_mfcc\net_audio_comb_mfcc.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
net = load('exp\audio_comb_mfcc\net_audio_comb_mfcc.mat') ;

% Load the test data
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\test\features\fea_audio_comb_mfcc_test.mat');
fea_audio_comb_mfcc_test=reshape(fea_audio_comb_mfcc_test,[1,1,size(fea_audio_comb_mfcc_test,1),size(fea_audio_comb_mfcc_test,2)]);
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\test\labels_audio_test.mat') ;

test_data = single(fea_audio_comb_mfcc_test);
test_data = test_data - net.meta.imageMean ;

% Apply the CNN
res = vl_simplenn(net, test_data) ;

% Visualize the results
test_pred=squeeze(res(end).x);
test_pred=test_pred';
csvwrite('exp\audio_comb_mfcc\result_audio_comb_mfcc.csv',test_pred);
diff = test_pred - labels_audio_test;
diff = abs(diff);
err_mean=mean(diff(:));
accuracy=1-err_mean;
fprintf('Test accuracy of audio regression net is %.6f%%.', accuracy);

end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.label(:,:,:,batch) ;
end
