dstFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\ext_frames';

curDir = pwd;

cd(dstFolder)

num = 3;
num_str = int2str(num);
mkdir(num_str);
dstFolder = fullfile(dstFolder, num_str);
cd(dstFolder);
mkdir('train');
mkdir('val');
mkdir('test');

interval = 25/num;

srcFilePattern = '*.jpg'; 

srcFolder_train = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\train\frames';
cd(srcFolder_train);
srcFileList = dir(srcFilePattern);
num_train = size(srcFileList, 1);
cd(curDir);
dstFolder_train = fullfile(dstFolder, 'train');
order_train = round(1 : interval : num_train);
for k=order_train
    baseFileName = srcFileList(k).name;
    filePath = fullfile(srcFolder_train, baseFileName);
    copyfile(filePath, dstFolder_train);
end

srcFolder_val = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\val\frames';
cd(srcFolder_val)
srcFileList = dir(srcFilePattern);
num_val = size(srcFileList, 1);
cd(curDir);
dstFolder_val = fullfile(dstFolder, 'val');
order_val = round(1 : interval : num_val);
for k=order_val
    baseFileName = srcFileList(k).name;
     filePath = fullfile(srcFolder_val, baseFileName);
    copyfile(filePath, dstFolder_val);
end

srcFolder_test = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\test\frames';
cd(srcFolder_test)
srcFileList = dir(srcFilePattern);
num_test = size(srcFileList, 1);
cd(curDir);
dstFolder_test = fullfile(dstFolder, 'test');
order_test = round(1 : interval : num_test);
for k=order_test
    baseFileName = srcFileList(k).name;
    filePath = fullfile(srcFolder_test, baseFileName);
    copyfile(filePath, dstFolder_test);
end
