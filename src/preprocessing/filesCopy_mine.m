srcFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\videos';
trainFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\train\videos';
valFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\val\videos';
testFolder = 'C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\test\videos';

curDir = pwd;

cd(srcFolder);
srcFileList = dir('*.mp4');
N=size(srcFileList,1);
perm_order = randperm(N);
for k = 1 : 3600
    baseFileName = srcFileList(perm_order(k)).name;
    copyfile(baseFileName, trainFolder);
end
for k = 3601 : 4800
    baseFileName = srcFileList(perm_order(k)).name;
    copyfile(baseFileName, valFolder);
end
for k = 4801 : 6000
    baseFileName = srcFileList(perm_order(k)).name;
    copyfile(baseFileName, testFolder);
end

cd(curDir);