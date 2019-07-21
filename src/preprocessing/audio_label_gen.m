function audio_label_gen(order, labelFolder)

load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\GroundTruth\gt.mat');

N=size(order,1);
labels=zeros(N,5);

for k = 1 : N
    baseFileName = order{k};
    videoNameIdx = strfind(baseFileName, '.wav');
    videoName = strcat(baseFileName(1: videoNameIdx),'mp4');
    index = ismember(gtVideoName, videoName);
    labels(k,:) = gtValue(index, :);
end

cd(labelFolder);
save('labels.mat', 'labels');

end