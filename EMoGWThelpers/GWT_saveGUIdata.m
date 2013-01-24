function S = GWT_saveGUIdata(pExampleName, GWT, Data, imgOpts, out_file)

out_dir = '/Users/emonson/Data/Fodava/EMoGWDataSets/';

% S will store/save variables & data for the GUI
S = struct();

%% Transfer over common entities (if they exist)

S.AmbientDimension = GWT.opts.AmbientDimension;
S.X = GWT.X;
S.CelWavCoeffs = Data.CelWavCoeffs;    
S.CelScalCoeffs = Data.CelScalCoeffs;

S.cp = GWT.cp;
S.num_nodes = length(GWT.cp);
S.LeafNodesImap(GWT.LeafNodes) = 1:length(GWT.LeafNodes);

S.IniLabels = GWT.IniLabels;
S.PointsInNet = GWT.PointsInNet;
S.NumberInNet = GWT.Sizes;
S.ScalFuns = GWT.ScalFuns;
S.WavBases = GWT.WavBases;
S.Centers = GWT.Centers;
S.Scales = GWT.Scales;
S.IsALeaf = GWT.isaleaf;
S.LeafNodes = GWT.LeafNodes;
S.EigenVecs = GWT.Graph.EigenVecs;
S.EigenVals = GWT.Graph.EigenVals;

% Flags
S.isImageData = false;
S.isTextData = false;
S.isCompressed = false;
S.isDownsampled = false;
S.hasLabels = false;
S.hasLabelMeanings = false;
S.hasLabelSetNames = false;
S.hasDocTitles = false;
S.hasDocFileNames = false;

% Images
if (isfield(imgOpts, 'imageData') && imgOpts.imageData)
  S.isImageData = true;
  S.imR = imgOpts.imR;
  S.imC = imgOpts.imC;
  
  if (isfield(imgOpts, 'isCompressed') && imgOpts.isCompressed)
    S.isCompressed = true;
    S.cm = imgOpts.cm;
    S.V = imgOpts.V;
  end
end

% Text
if (isfield(imgOpts, 'isTextData') && imgOpts.isTextData)
  S.isTextData = true;
  S.isImageData = false;
  S.Terms = imgOpts.Terms;
  
  if (isfield(imgOpts, 'DocTitles'))
    S.hasDocTitles = true;
    S.DocTitles = imgOpts.DocTitles;
  end
  if (isfield(imgOpts, 'DocFileNames'))
    S.hasDocFileNames = true;
    S.cat_labels = imgOpts.DocFileNames;
  end
end

% Labels
if (isfield(imgOpts, 'Labels'))
  S.hasLabels = true;
  S.Labels = imgOpts.Labels;
  
  if (isfield(imgOpts, 'LabelMeanings'))
    S.hasLabelMeanings = true;
    S.LabelMeanings = imgOpts.LabelMeanings;
  end
  if (isfield(imgOpts, 'LabelSetNames'))
    S.hasLabelSetNames = true;
    S.LabelSetNames = imgOpts.LabelSetNames;
  end
end

%% do example-dependent processing of GWT output data
fprintf('\nSaving %s data...\n', pExampleName);tic
switch pExampleName
    
  case 'MNIST_Digits'

  case 'Frey_faces'
    
  case 'Olivetti_faces'
    
  case 'ScienceNews'

  case '20NewsSubset1'

  case 'YaleB_Faces'
    
    fprintf(1,'Downsampling images\n');
    S.isDownsampled = true;

    d_fac = 0.1;
    imR2 = floor(imgOpts.imR*d_fac);
    imC2 = floor(imgOpts.imC*d_fac);
    N = size(GWT.X, 2);

    % Original (projected) images
    zz = zeros(imR2*imC2,N);
    for ii=1:N, 
        if mod(ii,100) == 0,
            fprintf(1,'%d\n',ii);
        end
        xx = S.V*S.X(:,ii) + S.cm';
        xx2 = reshape(xx,[S.imR S.imC]);
        yy = imresize(xx2, d_fac, 'nearest'); 
        zz(:,ii) = reshape(yy,[imR2*imC2 1]);
    end

    S.X = zz;

    % Centers
    fprintf(1,'Centers downsampling\n');
    zz = cell(size(S.Centers));
    for ii=1:size(S.Centers,2), 
        if mod(ii,10) == 0,
            fprintf(1,'%d\n',ii);
        end
        xx = S.V*S.Centers{ii} + S.cm';
        xx2 = reshape(xx,[S.imR S.imC]);
        yy = imresize(xx2, d_fac, 'nearest'); 
        zz{ii} = reshape(yy,[1 imR2*imC2]);
    end
    S.Centers = zz;

    % Wavelet bases
    fprintf(1,'Wavelet bases downsampling\n');
    bb = cell(size(S.WavBases));
    for nn=1:size(S.WavBases,2), 
        if mod(nn,10) == 0,
            fprintf(1,'%d\n',nn);
        end
        xx = S.V*S.WavBases{nn};
        zz = zeros(imR2*imC2,size(xx,2));
        for ii=1:size(xx,2), 
            im1 = xx(:,ii);
            im2 = reshape(im1,[S.imR S.imC]);
            yy = imresize(im2, d_fac, 'nearest'); 
            zz(:,ii) = reshape(yy,[imR2*imC2 1]);
        end
        bb{nn} = zz;
    end
    S.WavBases = bb;

    % Scaling functions
    fprintf(1,'Scaling functions downsampling\n');
    bb = cell(size(S.ScalFuns));
    for nn=1:size(S.ScalFuns,2), 
        if mod(nn,10) == 0,
            fprintf(1,'%d\n',nn);
        end
        xx = S.V*S.ScalFuns{nn};
        zz = zeros(imR2*imC2,size(xx,2));
        for ii=1:size(xx,2), 
            im1 = xx(:,ii);
            im2 = reshape(im1,[S.imR S.imC]);
            yy = imresize(im2, d_fac, 'nearest'); 
            zz(:,ii) = reshape(yy,[imR2*imC2 1]);
        end
        bb{nn} = zz;
    end
    S.ScalFuns = bb;

    S.imR = imR2;
    S.imC = imC2;
    % V = ones(size(X,2));
    % cm = zeros(size(X,2),1);

    % 90 deg CW rotation of original
    for ii = 1:size(S.Centers,2),
        yy = reshape(S.Centers{ii}, S.imR, S.imC);
        yy1 = permute(yy, [2 1]);
        yy2 = flipdim(yy1, 2);
        yy3 = reshape(yy2, 1, S.imR*S.imC);
        S.Centers{ii}(:) = yy3(:);
    end
    
    fprintf(1, 'Rotating images:\n');
    fprintf(1, '\tWavBases\n');
    for ii = 1:size(S.WavBases,2),
        for jj = 1:size(S.WavBases{ii},2),
            yy = reshape(S.WavBases{ii}(:,jj), S.imR, S.imC);
            yy1 = permute(yy, [2 1]);
            yy2 = flipdim(yy1, 2);
            yy3 = reshape(yy2, 1, S.imR*S.imC);
            S.WavBases{ii}(:,jj) = yy3(:);
        end
    end

    fprintf(1, '\tScalFuns\n');
    for ii = 1:size(S.ScalFuns,2),
        for jj = 1:size(S.ScalFuns{ii},2),
            yy = reshape(S.ScalFuns{ii}(:,jj), S.imR, S.imC);
            yy1 = permute(yy, [2 1]);
            yy2 = flipdim(yy1, 2);
            yy3 = reshape(yy2, 1, S.imR*S.imC);
            S.ScalFuns{ii}(:,jj) = yy3(:);
        end
    end

    fprintf(1, '\tX\n');
    for ii = 1:N,
        yy = reshape(S.X(:,ii), S.imR, S.imC);
        yy1 = permute(yy, [2 1]);
        yy2 = flipdim(yy1, 2);
        yy3 = reshape(yy2, S.imR*S.imC, 1);
        S.X(:,ii) = yy3(:);
    end

    tmp = S.imR;
    S.imR = S.imC;
    S.imC = tmp;
    S.isCompressed = false;
    S = rmfield(S, 'V');
    S = rmfield(S, 'cm');

end

%% Save data

fprintf(1, 'Saving data...\n');

save([out_dir out_file], 'S');

end