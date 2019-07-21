function dan_multi_scale(varargin)
% -------------------------------------------------------------------------
%prepare the data
% -------------------------------------------------------------------------

load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale\512\fea_video_dan_train.mat');
fea_video_dan_train=reshape(fea_video_dan_train,[1,1,size(fea_video_dan_train,1),size(fea_video_dan_train,2)]);
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale\labels_video_train.mat') ;
labels_video_train=labels_video_train';
labels_video_train=reshape(labels_video_train,[1,1,size(labels_video_train,1),size(labels_video_train,2)]);
num_train=size(fea_video_dan_train,4);

load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale\512\fea_video_dan_val.mat');
fea_video_dan_val=reshape(fea_video_dan_val,[1,1,size(fea_video_dan_val,1),size(fea_video_dan_val,2)]);
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale\labels_video_val.mat') ;
labels_video_val=labels_video_val';
labels_video_val=reshape(labels_video_val,[1,1,size(labels_video_val,1),size(labels_video_val,2)]);
num_val=size(fea_video_dan_val,4);

imdb.meta.sets={'train','val'};
imdb.images.data=single(cat(4, fea_video_dan_train, fea_video_dan_val));
imdb.images.label=single(cat(4, labels_video_train, labels_video_val));
%imdb.images.set=[ones(1,num_train), ones(1,num_val)+1];

% -------------------------------------------------------------------------
% initialize a CNN architecture
% -------------------------------------------------------------------------

run('C:\Study\Programming\Matlab\MatConvNet\matconvnet-1.0-beta23\matlab\vl_setupnn.m');


% -------------------------------------------------------------------------
% 5-fold cross validation
% -------------------------------------------------------------------------
imdb.images.set = int8(ones(1,num_train+num_val));%reset

if 0
    cv_idx = ml_kFoldCV_Idxs(size(imdb.images.set,2), 5);
    for cv_ite=1:1
        imdb.images.set(cv_idx{cv_ite})=2;%mark as validation

        net = dan_init(1024) ;
        % Add a loss (using a custom layer)
        net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;
        
        trainOpts.batchSize = 100 ;
        trainOpts.numEpochs = 100 ;
        trainOpts.continue = true ;
        trainOpts.gpus = [];
        trainOpts.learningRate = 0.001 ;
        trainOpts.expDir = sprintf('exp_cv\\%d',cv_ite) ;
        trainOpts.errorFunction = 'none' ;
        trainOpts = vl_argparse(trainOpts, varargin);

        % Take the average image out
        imageMean = mean(imdb.images.data(:,:,:,imdb.images.set==1),4) ;
        imdb.images.data = imdb.images.data - imageMean ;

        % Call training function in MatConvNet
        [net,~] = cnn_train(net, imdb, @getBatch, trainOpts) ;

        imdb.images.set(cv_idx{cv_ite})=1;%reset
    end
end


net = dan_init_2layers(1024) ;

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% -------------------------------------------------------------------------
% train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 4500;
trainOpts.continue = true ;
trainOpts.gpus = [];
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'exp_multiscale\dan_512' ;
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
save('exp_multiscale\dan_512\dan_512.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
net = load('exp_multiscale\dan_512\dan_512.mat') ;

% Load the test data
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale\512\fea_video_dan_test.mat');
fea_video_dan_test=reshape(fea_video_dan_test,[1,1,size(fea_video_dan_test,1),size(fea_video_dan_test,2)]);
load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale\labels_video_test.mat') ;

test_data = single(fea_video_dan_test);
test_data = test_data - net.meta.imageMean ;

% Apply the CNN
res = vl_simplenn(net, test_data) ;

% Visualize the results
test_pred=squeeze(res(end).x);
test_pred=test_pred';
csvwrite('exp_multiscale\dan_512\result_dan_512.csv',test_pred);
diff = test_pred - labels_video_test;
diff = abs(diff);
err_mean=mean(diff(:));
accuracy=1-err_mean;
fprintf('Test accuracy of DAN is %.3f%%.', accuracy);

end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.label(:,:,:,batch) ;
end
