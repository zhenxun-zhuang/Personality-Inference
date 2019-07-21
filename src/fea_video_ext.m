function fea_video_ext(srcFolder, dstFolder, width_resize)
parpool(2);

% Check to make sure that the source folder actually exists.  Warn user if it doesn't.
if ~isdir(srcFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', srcFolder);
  uiwait(warndlg(errorMessage));
  return;
end

run('C:\Study\Programming\Matlab\MatConvNet\matconvnet-1.0-beta23\matlab\vl_setupnn.m');
% Load the VGG-fece network
net = load('C:\Study\Research\Computer Vision\Introduction to Computer Vision\Project\Data\net\vgg-face_reduced.mat') ;
% Update the VGG-face net 
net = vl_simplenn_tidy(net) ;

%move the net to gpu
moveop = @(x) gpuArray(x) ;
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'convt', 'bnorm'}
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
      for f = {'weights', 'momentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          for j=1:numel(net.layers{l}.(f))
            net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
          end
        end
      end
    otherwise
      % nothing to do ?
  end
end

% Get a list of all files in the source folder with the desired file name pattern.
curDir = pwd;
cd(srcFolder);

srcFilePattern = '*.jpg'; 
srcFileList = dir(srcFilePattern);
train_data_num = size(srcFileList, 1);
fea_video_dan = zeros(1024, train_data_num);
fea_video_dan_plus = zeros(2048, train_data_num);

startT= tic;

avg_img = net.meta.normalization.averageImage;

parfor k = 1 : train_data_num
    
    if(rem(k,100)==1)
        progressBar(k, train_data_num, 'Generating VGG-Face features:', startT);
    end
    
    fileName = srcFileList(k).name;
    
    img=single(imread(fileName));
    img=imresize(img,[width_resize, NaN]);
    img=gpuArray(img);
    img=img - avg_img;% take the mean image out

    res = vl_simplenn(net, img);
    pool=res(32).x;
    relu=res(29).x;

    [fea_dan, fea_dan_plus]=desc_agg(pool, relu, width_resize);   
    fea_video_dan(:, k)=gather(fea_dan);
    fea_video_dan_plus(:, k)=gather(fea_dan_plus);
        
end

cd(dstFolder);
save('fea_video_dan.mat','fea_video_dan');
save('fea_video_dan_plus.mat','fea_video_dan_plus');

cd(curDir);

end

%selective descriptor aggregation
function [fea_dan, fea_dan_plus] = desc_agg(pool, relu, width_resize)

%activation map
pool_map = sum(pool, 3);
%dichotomize
pool_map_mean = mean(pool_map(:));
pool_mask = false(size(pool_map));
pool_mask(pool_map > pool_map_mean) = true;
%find largest connected component
cc_pool = bwconncomp(pool_mask);
numPixels_pool = cellfun(@numel,cc_pool.PixelIdxList);
[~, idx] = max(numPixels_pool);
pool_mask = false(size(pool_map));
pool_mask(cc_pool.PixelIdxList{idx}) = true;
%select region of interest
pool_dim = size(pool,1)*size(pool,2);
pool_res = reshape(pool, pool_dim, 512);
pool_res = pool_res(pool_mask==true, :);
%aggregate
pa=mean(pool_res,1);
pm=max(pool_res, [], 1);
%l2 normalization
pa_norm=norm(pa);
pa=pa./pa_norm;
pm_norm=norm(pm);
pm=pm./pm_norm;
%concatenation
fea_dan=[pa, pm]';

relu_height = size(relu,1);
relu_width  = size(relu,2);
relu_dim = relu_height*relu_width;
%activation map
relu_map = sum(relu, 3);
%dichotomize
relu_map_mean = mean(relu_map(:));
relu_mask = false(size(relu_map));
relu_mask(relu_map > relu_map_mean) = true;
%find largest connected component
cc_relu = bwconncomp(relu_mask);
numPixels_relu = cellfun(@numel,cc_relu.PixelIdxList);
[~, idx] = max(numPixels_relu);
relu_mask = false(size(relu_map));
relu_mask(cc_relu.PixelIdxList{idx}) = true;
pool_mask_up=imresize(pool_mask, [relu_height, relu_width]);
relu_mask=relu_mask & pool_mask_up;
if(~any(relu_mask(:)))%in case the two mask do not overlay at all
    relu_mask = pool_mask_up;
end
%select roi

relu_res = reshape(relu, relu_dim, 512);
relu_res = relu_res(relu_mask==true, :);
%aggregate
ra=mean(relu_res,1);
rm=max(relu_res, [], 1);
%l2 normalization
ra_norm=norm(ra);
ra=ra./ra_norm;
rm_norm=norm(rm);
rm=rm./rm_norm;
%concatenation
fea_dan_plus=[ra, rm]';
fea_dan_plus = [fea_dan; 0.5*fea_dan_plus];
%l2 normalization
fea_dan_plus_norm = norm(fea_dan_plus);
fea_dan_plus = fea_dan_plus./fea_dan_plus_norm;
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