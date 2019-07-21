function frame_label_gen(imgFolder, labelFolder)

% Check to make sure that the img folder and label file actually exists.  Warn user if it doesn't.
if ~isdir(imgFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', imgFolder);
  uiwait(warndlg(errorMessage));
  return;
end

% Get a list of all files in the folder with the desired file name pattern.
curDir = pwd;
cd(imgFolder);

imgFilePattern = '*.jpg'; % Change to whatever pattern you need.
imgFileList = dir(imgFilePattern);
N=size(imgFileList,1);

load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\GroundTruth\gt.mat');

labels=zeros(N,5);

startT= tic;
videoNamePre=' ';
index=0;
for k = 1 : N
    if(rem(k,100)==1)
        progressBar(k, N, 'Generating labels:', startT);
    end
    baseFileName = imgFileList(k).name;
    videoNameIdx = strfind(baseFileName, '.mp4');
    videoName = baseFileName(1: videoNameIdx+3);
    if(~strcmp(videoName, videoNamePre))
        videoNamePre=videoName;
        index = find(ismember(gtVideoName, videoName));
    end
    labels(k,:) = gtValue(index, :);
end

cd(labelFolder);
save('labels.mat', 'labels');

cd(curDir);

end

function progressBar(k,n, prefixMessage, startT)

if ~exist('prefixMessage', 'var') || isempty(prefixMessage)
    prefixMessage = 'Progress';
end;

nDigit = length(sprintf('%d',n));

if exist('startT', 'var') && ~isempty(startT)
    format = sprintf('%%%dd/%d (%%6.2f%%%%), elapse time: %7.1fs', nDigit, n, toc(startT));
    delFormat = repmat('\b', 1, 2*nDigit+11+23);    
else
    format = sprintf('%%%dd/%d (%%6.2f%%%%)', nDigit, n);
    delFormat = repmat('\b', 1, 2*nDigit+11);
end

if k==1
    fprintf('\n');
    fprintf([prefixMessage ' ' format], k, 100*k/n);
elseif k == n
    fprintf([delFormat, format, '\n'], k, 100*k/n);
else
    fprintf([delFormat, format], k, 100*k/n);
end

end