% Rotating subsampled images for GUI
% GUI rotates Matlab images 90 deg CCW, so
% need to pre-rotate images here in prep

clear all;
close all;

load('/Users/emonson/Data/Fodava/EMoGWDataSets/yaleB_pca200_20100103.mat');

% 90 deg CW rotation of original
for ii = 1:size(Centers_down,2),
    yy = reshape(Centers_down{ii}, imR_down, imC_down);
    yy1 = permute(yy, [2 1]);
    yy2 = flipdim(yy1, 2);
    yy3 = reshape(yy2, 1, imR_down*imC_down);
    Centers_down{ii}(:) = yy3(:);
end

fprintf(1, 'WavBases\n');
for ii = 1:size(WavBases_down,2),
    for jj = 1:size(WavBases_down{ii},2),
        yy = reshape(WavBases_down{ii}(:,jj), imR_down, imC_down);
        yy1 = permute(yy, [2 1]);
        yy2 = flipdim(yy1, 2);
        yy3 = reshape(yy2, 1, imR_down*imC_down);
        WavBases_down{ii}(:,jj) = yy3(:);
    end
end

fprintf(1, 'ScalFuns\n');
for ii = 1:size(ScalFuns_down,2),
    for jj = 1:size(ScalFuns_down{ii},2),
        yy = reshape(ScalFuns_down{ii}(:,jj), imR_down, imC_down);
        yy1 = permute(yy, [2 1]);
        yy2 = flipdim(yy1, 2);
        yy3 = reshape(yy2, 1, imR_down*imC_down);
        ScalFuns_down{ii}(:,jj) = yy3(:);
    end
end

fprintf(1, 'X\n');
for ii = 1:size(X_down,1),
    yy = reshape(X_down(ii,:), imR_down, imC_down);
    yy1 = permute(yy, [2 1]);
    yy2 = flipdim(yy1, 2);
    yy3 = reshape(yy2, 1, imR_down*imC_down);
    X_down(ii,:) = yy3(:);
end

tmp = imR_down;
imR_down = imC_down;
imC_down = tmp;

figure; imagesc(reshape(Centers_down{1}, imR_down, imC_down));
figure; imagesc(reshape(WavBases_down{1}(:,1), imR_down, imC_down));
figure; imagesc(reshape(ScalFuns_down{1}(:,1), imR_down, imC_down));

save('/Users/emonson/Data/Fodava/EMoGWDataSets/yaleB_pca200_20110329.mat');
