curDir = pwd;

cd('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\ext_frames\3\test');
img_name = dir('*.jpg');
cd('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\test\videos');
video_name = dir('*.mp4');

cd(curDir);

result = zeros(1200,5);

img_num = size(img_name,1);

videoNamePre=' ';
index = 0;
num = 0;
for i=1:img_num
    baseFileName = img_name(i).name;
    videoNameIdx = strfind(baseFileName, '.mp4');
    videoName = baseFileName(1: videoNameIdx+3);
    if(~strcmp(videoName, videoNamePre))
        if (index~=0)
            result(index,:) = result(index,:) / num;
            num = 0;
        end
        videoNamePre=videoName;
        index = find(arrayfun(@(n) strcmp(video_name(n).name, videoName), 1:numel(video_name)));
    end
    result(index,:) = result(index,:) + result_video(i,:);
    num = num + 1;
end

result(index,:) = result(index,:) / num;