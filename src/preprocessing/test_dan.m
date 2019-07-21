img=imresize(im, [384, NaN]);
img=img - avg_img;% take the mean image out

res = vl_simplenn(net, img);
pool=res(32).x;

%activation map
pool_map = sum(pool, 3);
%dichotomize
pool_map_mean = mean(pool_map(:));
pool_mask = false(size(pool_map));
pool_mask(pool_map >= pool_map_mean) = true;
%find largest connected component
cc_pool = bwconncomp(pool_mask);
numPixels_pool = cellfun(@numel,cc_pool.PixelIdxList);
[~, idx] = max(numPixels_pool);
pool_mask = false(size(pool_map));
pool_mask(cc_pool.PixelIdxList{idx}) = true;

pool_mask_ups = imresize(pool_mask);
a= iminfuse(pool_mask_ups, im);
imagesc(a);