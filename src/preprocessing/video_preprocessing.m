function video_preprocessing(srcFolder, dstFolder, numExtFramesPerVideo)

% Check to make sure that the source folder actually exists.  Warn user if it doesn't.
if ~isdir(srcFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', srcFolder);
  uiwait(warndlg(errorMessage));
  return;
end

% Check to make sure that the destination folder actually exists.  Create it if it doesn't.
if ~isdir(dstFolder)
    mkdir(dstFolder);
end

% Get a list of all files in the folder with the desired file name pattern.
curDir = pwd;
cd(srcFolder);

srcFilePattern = '*.mp4'; % Change to whatever pattern you need.
srcFileList = dir(srcFilePattern);
N=size(srcFileList,1);
startT= tic;
for k = 3349 : N
    if(rem(k,10)==1)
        progressBar(k, N, 'Extracting frames from videos:', startT);
    end
    baseFileName = srcFileList(k).name;
    extractFrames(baseFileName, numExtFramesPerVideo, dstFolder);
end

cd(curDir);

end

function extractFrames(srcFile, numExtFramesPerVideo, dstFolder)

v = VideoReader(srcFile);

curFrameCount = 0;
extFrameCount = 1;
totalFrames = v.duration*v.FrameRate;
interval = totalFrames/numExtFramesPerVideo;
while hasFrame(v)
    curFrame = readFrame(v);
    curFrameCount = curFrameCount+1;
    if (round(extFrameCount * interval) == curFrameCount)
        dstFileName = sprintf('%s_%d.jpg', srcFile, extFrameCount);
        fullDstFileName = fullfile(dstFolder, dstFileName);
        imwrite(curFrame, fullDstFileName);
        extFrameCount = extFrameCount + 1;
    end
end
    
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
