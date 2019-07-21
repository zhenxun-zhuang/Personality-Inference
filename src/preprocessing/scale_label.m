dstFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\multiscale';
cd(dstFolder)

num = 3;
interval = 25/num;

num_train = size(labels_frame_train,1);
order_train = round(1 : interval : num_train);
labels_video_train = labels_frame_train(order_train, :);
save('labels_video_train.mat', 'labels_video_train')

num_val = size(labels_frame_val,1);
order_val = round(1 : interval : num_val);
labels_video_val = labels_frame_val(order_val, :);
save('labels_video_val.mat', 'labels_video_val')

num_test = size(labels_frame_test,1);
order_test = round(1 : interval : num_test);
labels_video_test = labels_frame_test(order_test, :);
save('labels_video_test.mat', 'labels_video_test')