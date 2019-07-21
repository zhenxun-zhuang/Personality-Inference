dstFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\num_ext_frames';
cd(dstFolder)

num = 1;
num_str = int2str(num);
mkdir(num_str);
cd(num_str);

interval = 100/num;

num_train = size(labels_train,1);
order_train = round(1 : interval : num_train);
fea_video_train = vgg_fea_dan_train(:, order_train);
save('fea_video_train.mat', 'fea_video_train');
labels_video_train = labels_train(order_train, :);
save('labels_video_train.mat', 'labels_video_train')

num_val = size(labels_val,1);
order_val = round(1 : interval : num_val);
fea_video_val = vgg_fea_dan_val(:, order_val);
save('fea_video_val.mat', 'fea_video_val');
labels_video_val = labels_val(order_val, :);
save('labels_video_val.mat', 'labels_video_val')

num_test = size(labels_test,1);
order_test = round(1 : interval : num_test);
fea_video_test = vgg_fea_dan_test(:, order_test);
save('fea_video_test.mat', 'fea_video_test');
labels_video_test = labels_test(order_test, :);
save('labels_video_test.mat', 'labels_video_test')

