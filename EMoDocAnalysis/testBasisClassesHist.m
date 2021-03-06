% clear all;
% baseName = 'X20_042709b';
% 
% cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
% fprintf(1,'Loading data set\n');
%     load([baseName '.mat']);

% 20 newsgroups TFIDF nltk tokenize
clear all;
save_dir = '/Users/emonson/Data/Fodava/EMoDocMatlabData';
save_name = 'X20_042709b.mat';

cd(save_dir);
fprintf(1,'Loading data set\n');
load(save_name);

pts = dlmread([save_name(1:(end-4)) '_pts.csv'],' ');

scale = 3;
nF = 40;    % Number of basis functions displayed
sD = 5;     % Subplot count in down direction in figure window
sA = 8;     % Subplot count in across direction

HIST = 1;               % Display "histogram" bars
CLAS = 1;               % Display class graph
BASI = 1;   BATI = 1;   % Display basis functions plots & T value titles
EIGS = 1;               % Display eigenfunction plots

% Clean up classes if need to...
classes(classes(:,1)==0,:) = [];

if(HIST),
    figure;
    classHist = zeros(nF,8);
    extBasis = G.Tree{scale,1}.ExtBasis;
    for ii = 1:nF,
        extVec = full(extBasis(:,ii));

        for jj = 1:8,
            classHist(ii,jj) = sum((classes(:,1)==jj).*(extVec.^2));
        end;
    end;
    bar(classHist,'stacked');
    colormap(brewerDark1(8));
    caxis([0.5 8.5]);
    axis([0 nF+1 -0.05 1.15]); 
    colorbar;
    for ii = 1:nF,
        label = sprintf('%d',classes(G.Tree{scale,1}.ExtIdxs(ii),1));
        text(ii-0.1,1.05,label);
    end;
    set(gca,'TickLength',[0.005 0.005]);
end;

if(CLAS),
    figure; 
    scatter(pts(:,1),pts(:,2),80,classes(:,1),'filled');
    colormap(brewerDark1(8));
    caxis([0.5 8.5]);

    colorbar;
end;

if(BASI),
    figure; 
    for ii=1:nF, 
        subplot(sD,sA,ii); 
        scatter(pts(:,1),pts(:,2),30,G.Tree{scale,1}.ExtBasis(:,ii),'filled'); 
        if(BATI),
            titleString = sprintf('%d � %3.2f � %d',ii, G.Tree{scale,1}.T{1}(ii,ii), classes(G.Tree{scale,1}.ExtIdxs(ii),1));
            % titleString = sprintf('%d � %3.2f',ii, entropy(full(G.Tree{scale,1}.ExtBasis(:,ii)).^2));
            title(titleString);
        end;
        colormap(map3); 
        balanceColor; 
    end;
end;

if(EIGS),
    figure; 
    for ii=1:nF, 
        subplot(sD,sA,ii); 
        scatter(pts(:,1),pts(:,2),30,G.EigenVecs(:,ii),'filled'); 
        colormap(map3); 
        balanceColor; 
    end;
end;
