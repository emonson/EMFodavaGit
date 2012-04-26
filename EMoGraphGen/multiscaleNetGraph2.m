% pFileName = 'mutliscalenetworkgraph.bmp';
% 
% fprintf('Loading image...');
% try
%     % Load the file containing the description of the rooms environment
%     lImage = imread(pFileName);        
% catch
%     fprintf('\n Error: could not load the description file %s!',pFileName);
%     return;
% end;
% fprintf('done.\n');
% 
% lSumImage = sum(lImage,3);
% [lI,lJ]   = find(lSumImage==0);
% 
% % Create the data set from the picture
% lX = [lI';lJ'];
% lX_s = lX(:,1:3:size(lX,2));
% lX_s = [0,-1;1,0]*lX_s;
% 
% % Create a diffusion on the data set
% T = MakeDiffusion( lX_s, 25, struct('Normalization','smarkov') );
% 
% % Compute the multiscale decomposition
% lOpts = [];
% lOpts.Wavelets = false;
% %lOpts.GSOptions.StopDensity = 0.5;
% Tree = DWPTree (T, 10, 1e-2, lOpts);

% Display the multiscale decomposition
% Conversion of equivalents from original Fodava proposal
% multiscaleNetGraph.m
% Tree = G.Tree;
% lX_s = G.X;
% lScalingFcns{jj}(:,k) = G.Tree{jj,1}.ExtBasis(:,kk);

% Compute the center of masses from the scaling functions
% clear vCenters;
figure;
set(gcf, 'Color', 'w');

for j = 1:size(G.Tree,1),
    % for k = min([ceil(size(Tree{j,1}.ExtBasis,2)/5),20]):-1:1,
    lMainIdxs = find(abs(diag(G.Tree{j,1}.T{1}))>0.3);
    for k = size(lMainIdxs):-1:1,
        % vCenters{j}(:,k) = lX_s*(abs(Tree{j,1}.ExtBasis(:,k))/sum(abs(Tree{j,1}.ExtBasis(:,k))));
        vCenters{j}(:,k) = G.X*(G.Tree{j,1}.ExtBasis(:,k)/sum(G.Tree{j,1}.ExtBasis(:,k)));
    end;
end;

for j = 4:size(G.Tree,1),
    fprintf(1,'Tree: %d\n',j);
    subplot(4,4,j-3);
    if(j<4)
        plot(G.X(1,:),G.X(2,:),'.','MarkerSize',8,'Color',[0.4 0.4 0.4]);
    else
        plot(G.X(1,:),G.X(2,:),'.','MarkerSize',5,'Color',[0.8 0.8 0.8]);
        hold on;
        plot(vCenters{j-3}(1,:),vCenters{j-3}(2,:),'r.','MarkerSize',8,'Color',[0.4,0.4,0.4]);
    end;
    hold on;
    title(sprintf('Level %d',j));
    axis image;
    set(gca,'Visible','off');
    %gplot(full(double(abs(Tree{j,1}.T{1}(1:size(vCenters{j},2),1:size(vCenters{j},2)))>1e-6)),vCenters{j}'*[0,-1;1,0]);
    % Now plot the edges
    lMaxSim = max(max(abs(G.Tree{j,1}.T{1}(1:size(vCenters{j},2),1:size(vCenters{j},2))-diag(diag(G.Tree{j,1}.T{1}(1:size(vCenters{j},2),1:size(vCenters{j},2)))))));
    for e1 = 1:size(vCenters{j},2),
        fprintf(1,'\te1=%d/%d\n', e1, size(vCenters{j},2));
        for e2 = e1+1:size(vCenters{j},2),
            % fprintf(1,'\t%d, %d, %f\n', e1, e2, (1/lMaxSim)*abs(Tree{j,1}.T{1}(e1,e2)) );
            if abs(G.Tree{j,1}.T{1}(e1,e2))>=0.01,
                l=line( [vCenters{j}(1,e1),vCenters{j}(1,e2)],[vCenters{j}(2,e1),vCenters{j}(2,e2)] );            
                set(l,'LineWidth',(5/lMaxSim)*abs(G.Tree{j,1}.T{1}(e1,e2)));
                set(l,'Color',[(0.2/lMaxSim)*abs(G.Tree{j,1}.T{1}(e1,e2)),(0.2/lMaxSim)*abs(G.Tree{j,1}.T{1}(e1,e2)),(0.9/lMaxSim)*abs(G.Tree{j,1}.T{1}(e1,e2))]);
            end;
        end;
    end;
    plot(vCenters{j}(1,:),vCenters{j}(2,:),'ro','LineWidth',1.5,'MarkerSize',5,'Color',[0.8,0.1,0.1]);
end;