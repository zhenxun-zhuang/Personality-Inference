dstFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\num_ext_frames\10';
cd(dstFolder)

interval = 10;

num_train = size(vgg_fea_dan_plus_train,2);
order_train = round(1 : interval : num_train);
fea_video_train_plus = vgg_fea_dan_plus_train(:, order_train);
save('fea_video_train_plus.mat', 'fea_video_train_plus');

num_val = size(vgg_fea_dan_plus_val,2);
order_val = round(1 : interval : num_val);
fea_video_val_plus = vgg_fea_dan_plus_val(:, order_val);
save('fea_video_val_plus.mat', 'fea_video_val_plus');

num_test = size(vgg_fea_dan_plus_test,2);
order_test = round(1 : interval : num_test);
fea_video_test_plus = vgg_fea_dan_plus_test(:, order_test);
save('fea_video_test_plus.mat', 'fea_video_test_plus');