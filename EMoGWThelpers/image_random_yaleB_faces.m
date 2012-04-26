N = size(gW.X, 1);
rand_picks = randi(N,[1 20]);
imR = 480;
imC = 640;

figure;
for ii = 1:20, 
    pick = rand_picks(ii);
    xx = gW.X(pick,:)*Data.V' + Data.Xmean;
    xx2 = reshape(xx,[imR imC]);
    subplot(4,5,ii);
    imagesc(xx2);
    colormap(gray);
    axis image;
    axis off;
end
