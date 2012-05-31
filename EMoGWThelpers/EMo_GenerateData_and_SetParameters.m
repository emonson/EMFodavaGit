function [X, GWTopts, imgOpts] = GenerateData_and_SetParameters(pExampleName)

%% set GWT parameters
GWTopts = struct();

% The following thresholds are used in the code construct_GMRA.m
GWTopts.threshold1 = 1e-5;  % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.threshold2 = .1;    % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}

% whether to use best approximations
GWTopts.addTangentialCorrections = false;

% whether to sparsify the scaling functions and wavelet bases
GWTopts.sparsifying = false;
GWTopts.sparsifying_method = 'ksvd'; % or 'spams'

% whether to split the wavelet bases into a common intersection and
% children-specific parts
GWTopts.splitting = false;

% METIS parameters
GWTopts.knn = 30;
GWTopts.knnAutotune = 20;
GWTopts.smallestMetisNet = 10;

% whether to output time
GWTopts.verbose = 1;

% method for shrinking the wavelet coefficients
GWTopts.shrinkage = 'hard';

% whether to avoid using the scaling functions at the leaf nodes and
% instead using the union of their wavelet bases and the scaling functions
% at the parents
GWTopts.avoidLeafnodePhi = true;

% whether to merge the common part of the wavelet subspaces
% associated to the children into the scaling function of the parent.
GWTopts.mergePsiCapIntoPhi  = false;

imgOpts = struct();

% Flags
imgOpts.isImageData = false;
imgOpts.isTextData = false;
imgOpts.isCompressed = false;
imgOpts.isDownsampled = false;
imgOpts.hasLabels = false;
imgOpts.hasLabelMeanings = false;
imgOpts.hasLabelSetNames = false;
imgOpts.hasDocTitles = false;
imgOpts.hasDocFileNames = false;

%% create data, and set additional parameters
fprintf('\nGenerating/loading %s data...', pExampleName);tic
switch pExampleName

  case 'MNIST_Digits'

    % generate the dataset
    dataset = struct();
    dataset.N = 1000;
    dataset.digits = [1 2 3];
    dataset.projectionDimension = 0;

    [X0,GraphDiffOpts,NetsOpts,Labels] = GenerateDataSets( 'BMark_MNIST', ...
        struct('NumberOfPoints',dataset.N,'AutotuneScales',false,'MnistOpts',struct('Sampling', 'FirstN', 'QueryDigits',dataset.digits, 'ReturnForm', 'vector'))); %#ok<ASGLU>

    % Rotate images for GUI
    yy = reshape(X0,28,28,[]);
    yy1 = permute(yy,[2 1 3]);
    yy2 = flipdim(yy1,2);
    X0 = reshape(yy2,784,[]);
    clear('yy','yy1','yy2');

    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 28;
    imgOpts.imC = 28;
    imgOpts.Labels = Labels'; % Should be nCats x nPoints

    if dataset.projectionDimension>0 && dataset.projectionDimension<imgOpts.imR*imgOpts.imC,
        imgOpts.X0 = X0;
        imgOpts.cm = mean(X0,2);
        X = X0 - repmat(imgOpts.cm,1,dataset.N);
        %     [U,S,V] = svd(X,0);
        [U,S,V] = randPCA(X,dataset.projectionDimension);
        X = U.*repmat(diag(S)', dataset.N, 1);
        GWTopts.errorType = 'absolute';
        imgOpts.V = U;
        imgOpts.isCompressed = true;
    else
        X = X0; clear X0;
        imgOpts.isCompressed = false;
    end;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; % if 0, then determine locally adaptive dimensions using the following fields:
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = .05; % only for leaf nodes

  case 'Medical12images'

    cd('/Users/emonson/Data/Ronak');

    % I = csvread('Medical12Classes_data.csv');
    Labels = csvread('Medical12Classes_classes.csv');
    fid = fopen('Medical12Classes_filenames.csv');
    C = textscan(fid,'%s');
    filenames = C{1};
    fclose(fid);
    
    % randomize order
    idxs = randperm(length(filenames));
    filenames = filenames(idxs);
    Labels = Labels(idxs);
    
    cd('Medical12Classes');
    tmp = imread(filenames{1});
    [imR,imC] = size(tmp);

    dataset = struct();
    dataset.N = length(filenames);
    dataset.projectionDimension = 0;

    X0 = zeros(imR*imC, length(filenames));
    for ii = 1:length(filenames),
        tmp = imread(filenames{ii});
        X0(:,ii) = double(reshape(tmp, imR*imC, 1));
    end;

    % Rotate images for GUI
%     yy = reshape(X0,28,28,[]);
%     yy1 = permute(yy,[2 1 3]);
%     yy2 = flipdim(yy1,2);
%     X0 = reshape(yy2,784,[]);
%     clear('yy','yy1','yy2');

    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = imR;
    imgOpts.imC = imC;
    imgOpts.Labels = Labels'; % Should be nCats x nPoints

    if dataset.projectionDimension>0 && dataset.projectionDimension<imgOpts.imR*imgOpts.imC,
        imgOpts.X0 = X0;
        imgOpts.cm = mean(X0,2);
        X = X0 - repmat(imgOpts.cm,1,dataset.N);
        %     [U,S,V] = svd(X,0);
        [U,S,V] = randPCA(X, dataset.projectionDimension);
        X = U.*repmat(diag(S)', dataset.N, 1);
        GWTopts.errorType = 'absolute';
        imgOpts.V = U;
        imgOpts.isCompressed = true;
    else
        X = X0; clear X0;
        imgOpts.isCompressed = false;
    end;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; % if 0, then determine locally adaptive dimensions using the following fields:
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = .05; % only for leaf nodes

  case 'Medical12Sift'

    cd('/Users/emonson/Data/Ronak');

    % Data comes in as NxD (540x28)
    I = csvread('Medical12Classes_data.csv');
    Labels = csvread('Medical12Classes_classes.csv');
     
    % randomize order
    idxs = randperm(size(I,1));
    Labels = Labels(idxs);
    I = I(idxs,:);
    
    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 4;
    imgOpts.imC = 7;
    imgOpts.Labels = Labels'; % Should be nCats x nPoints

    X = I';
    imgOpts.isCompressed = false;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; % if 0, then determine locally adaptive dimensions using the following fields:
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = .05; % only for leaf nodes

  case 'CorelImages'

    cd('/Users/emonson/Data/Ronak');

    % I = csvread('Medical12Classes_data.csv');
    Labels = csvread('ImagesCorel_classes.csv');
    fid = fopen('ImagesCorel_filenames.csv');
    C = textscan(fid,'%s');
    filenames = C{1};
    fclose(fid);
    
    % randomize order
    idxs = randperm(length(filenames));
    filenames = filenames(idxs);
    Labels = Labels(idxs);
    
    cd('ImagesCorel');
    tmp = imread(filenames{1});
    [imR,imC,c] = size(tmp);
    if (imR < imC),
        tmp = permute(tmp,[2 1 3]);
    end;
    % Trying to compensate for 3-color images
    [imR,imC,c] = size(tmp);
    imC = 3*imC;

    dataset = struct();
    dataset.N = length(filenames);
    dataset.projectionDimension = 0;

    X0 = zeros(imR*imC, length(filenames));
    for ii = 1:length(filenames),
        tmp = imread(filenames{ii});
        [r,c,cc] = size(tmp);
        if (r < c),
            tmp = permute(tmp,[2 1 3]);
        end;
       X0(:,ii) = double(reshape(tmp, imR*imC, 1));
    end;

    % Rotate images for GUI
%     yy = reshape(X0,28,28,[]);
%     yy1 = permute(yy,[2 1 3]);
%     yy2 = flipdim(yy1,2);
%     X0 = reshape(yy2,784,[]);
%     clear('yy','yy1','yy2');

    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = imR;
    imgOpts.imC = imC;
    imgOpts.Labels = Labels'; % Should be nCats x nPoints

    if dataset.projectionDimension>0 && dataset.projectionDimension<imgOpts.imR*imgOpts.imC,
        imgOpts.X0 = X0;
        imgOpts.cm = mean(X0,2);
        X = X0 - repmat(imgOpts.cm,1,dataset.N);
        %     [U,S,V] = svd(X,0);
        [U,S,V] = randPCA(X,dataset.projectionDimension);
        X = U.*repmat(diag(S)', dataset.N, 1);
        GWTopts.errorType = 'absolute';
        imgOpts.V = U;
        imgOpts.isCompressed = true;
    else
        X = X0; clear X0;
        imgOpts.isCompressed = false;
    end;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; % if 0, then determine locally adaptive dimensions using the following fields:
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = .05; % only for leaf nodes

  case 'CorelSift'

    cd('/Users/emonson/Data/Ronak');

    % Data comes in as NxD (1000x150)
    I = csvread('ImagesCorel_data.csv');
    Labels = csvread('ImagesCorel_classes.csv');
     
    % randomize order
    idxs = randperm(size(I,1));
    Labels = Labels(idxs);
    I = I(idxs,:);
    
    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 10;
    imgOpts.imC = 15;
    imgOpts.Labels = Labels'; % Should be nCats x nPoints

    X = I';
    imgOpts.isCompressed = false;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; % if 0, then determine locally adaptive dimensions using the following fields:
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = .05; % only for leaf nodes

  case 'YaleB_Faces'

    load YaleB_PCA
    X = V*S; %#ok<NODEF>
    X = X';
    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 480;
    imgOpts.imC = 640;

    imgOpts.cm =  Imean';
    imgOpts.V = U; %#ok<NODEF>
    imgOpts.isCompressed = true;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; 
    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.05; % only for leaf nodes
    
    % GUI info
    imgOpts.hasLabels = true;
    imgOpts.Labels = Labels; % % Should be nCats x nPoints
    
    imgOpts.hasLabelSetNames = true;
    imgOpts.LabelSetNames = {'illumination', 'pose', 'subject'};

  case 'croppedYaleB_Faces'

    load extendedYaleB_crop_SVD
    dataset.projectionDimension = 500;
    %X = V(:,1:dataset.projectionDimension); %#ok<NODEF>
    X = bsxfun(@times, V(:,1:dataset.projectionDimension), (diag(S(1:dataset.projectionDimension,1:dataset.projectionDimension)))'); %#ok<NODEF>
    X = X';
    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 192;
    imgOpts.imC = 168;

    %imgOpts.Labels = Labels; %#ok<NODEF>
    imgOpts.cm =  center';
    %imgOpts.V = bsxfun(@times, U(:,1:dataset.projectionDimension), (diag(S(1:dataset.projectionDimension,1:dataset.projectionDimension)))'); %#ok<NODEF>
    imgOpts.V = U(:,1:dataset.projectionDimension); %#ok<NODEF>
    imgOpts.isCompressed = true;

    % GWT parameters that need to be set separately
    %GWTopts.ManifoldDimension = 4; 
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = 0.05; % only for leaf nodes

  case 'Frey_faces'
    
    load('frey_rawface.mat');
    X = double(fliplr(ff'))';

    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 20;
    imgOpts.imC = 28;

    % GWT parameters that need to be set separately
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = 0.05; % only for leaf nodes

  case 'Olivetti_faces'
    
    load('olivettifaces.mat');
    X0 = double(fliplr(faces'));
    imR = 64;
    imC = 64;
    % Rotate images for GUI
    yy = reshape(X0,[],imR,imC);
    yy1 = permute(yy,[1 3 2]);
    yy2 = flipdim(yy1,2);
    X = reshape(yy2,[],imR*imC)';
  
    % image parameters
    imgOpts.imageData = true;
    imgOpts.imR = 64;
    imgOpts.imC = 64;

    % GWT parameters that need to be set separately
    GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.errorType = 'relative';
    GWTopts.precision  = 0.05; % only for leaf nodes
    
    imgOpts.hasLabels = true;
    imgOpts.hasLabelSetNames = true;

    % 40 people, 10 images each, in order
    imgOpts.Labels = reshape(repmat(0:39,10,1),[],1)';
    imgOpts.LabelSetNames = {'subject'};

  case 'ScienceNews'

    load X20

    classes_orig = classes;
    classes(classes_orig(:,1)==0,:) = [];

    ff = fopen('sn_titles.txt');
    xx = textscan(ff, '%f%s', 'Delimiter', '\t');
    fclose(ff);

    file_names = xx{1};
    titles_orig = xx{2};

    ii = 1;
    titles = cell(size(classes(:,2)));

    for tt = classes(:,2)',
        titles{ii} = titles_orig{file_names == tt};
        ii = ii + 1;
    end

    fid = fopen('sn_LabelsMeaning.txt');
    articlegroups = textscan(fid,'%d = %s', 'Delimiter', '' );
    fclose(fid);

    X(classes_orig(:,1)==0,:) = [];
    X = X';
    
    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocTitles = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.DocTitles = titles;
    imgOpts.Terms = dict;
    imgOpts.Labels = classes(:,1)';
    imgOpts.DocFileNames = classes(:,2);
    imgOpts.LabelMeanings = articlegroups{2}';
    imgOpts.LabelSetNames = {'scientific discipline'};
    imgOpts.isTextData = true;
    
    % DEBUG: Exclude some outliers
    outliers = [358, 672, 731, 861];
    X(:,outliers) = [];
    imgOpts.Labels(outliers) = [];
    imgOpts.DocTitles(outliers) = [];
    imgOpts.DocFileNames(outliers) = [];

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.85 0.75]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

   case 'ScienceNewsTFIDF'

    load X20

    classes_orig = classes;
    classes(classes_orig(:,1)==0,:) = [];

    ff = fopen('sn_titles.txt');
    xx = textscan(ff, '%f%s', 'Delimiter', '\t');
    fclose(ff);

    file_names = xx{1};
    titles_orig = xx{2};

    ii = 1;
    titles = cell(size(classes(:,2)));

    for tt = classes(:,2)',
        titles{ii} = titles_orig{file_names == tt};
        ii = ii + 1;
    end

    fid = fopen('sn_LabelsMeaning.txt');
    articlegroups = textscan(fid,'%d = %s', 'Delimiter', '' );
    fclose(fid);

    X(classes_orig(:,1)==0,:) = [];
    I = X';

    % try moving back to counts instead of normalized freq/doc
    for ii = 1:size(I,2), 
      I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); 
    end;
    tdm = round(I);

    % calculate TFIDF (std) normalization for word counts
    nkj = sum(tdm,1)';      % how many terms in each document
    D = size(tdm,2);        % number of documents
    df = sum(tdm>0,2);      % number of documents each term shows up in
    idf = log(D./(1+df));   % the 1+ is common to avoid divide-by-zero

    [ii,jj,vv] = find(tdm);
    vv_norm = (vv./nkj(jj)).*idf(ii);

    tdm_norm = sparse(ii,jj,vv_norm);
    X = full(tdm_norm);

    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocTitles = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.DocTitles = titles;
    imgOpts.Terms = dict;
    imgOpts.Labels = classes(:,1)';
    imgOpts.DocFileNames = classes(:,2);
    imgOpts.LabelMeanings = articlegroups{2}';
    imgOpts.LabelSetNames = {'scientific discipline'};
    imgOpts.isTextData = true;

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.85 0.75]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

   case 'ScienceNewsCounts'

    load X20

    classes_orig = classes;
    classes(classes_orig(:,1)==0,:) = [];

    ff = fopen('sn_titles.txt');
    xx = textscan(ff, '%f%s', 'Delimiter', '\t');
    fclose(ff);

    file_names = xx{1};
    titles_orig = xx{2};

    ii = 1;
    titles = cell(size(classes(:,2)));

    for tt = classes(:,2)',
        titles{ii} = titles_orig{file_names == tt};
        ii = ii + 1;
    end

    fid = fopen('sn_LabelsMeaning.txt');
    articlegroups = textscan(fid,'%d = %s', 'Delimiter', '' );
    fclose(fid);

    X(classes_orig(:,1)==0,:) = [];
    I = X';

    % try moving back to counts instead of normalized freq/doc
    for ii = 1:size(I,2), 
      I(:,ii) = I(:,ii)./min(I(I(:,ii)>0,ii)); 
    end;
    X = round(I);

    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocTitles = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.DocTitles = titles;
    imgOpts.Terms = dict;
    imgOpts.Labels = classes(:,1)';
    imgOpts.DocFileNames = classes(:,2);
    imgOpts.LabelMeanings = articlegroups{2}';
    imgOpts.LabelSetNames = {'scientific discipline'};
    imgOpts.isTextData = true;

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.85 0.75]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

case '20NewsSubset1'

    load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub1train_TFcorr_111809.mat');

    X = full(tdm_norm);

    % GUI data
    imgOpts.isTextData = true;
    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.Terms = terms';
    imgOpts.Labels = labels;
    imgOpts.LabelMeanings = newsgroups;
    imgOpts.LabelSetNames = {'newsgroup'};
    imgOpts.DocFileNames = names';

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.9 0.8]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

 case '20NewsSubset2tf'

    load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub2train_TFcorr_111809.mat');

    X = full(tdm_norm);

    % GUI data
    imgOpts.isTextData = true;
    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.Terms = terms';
    imgOpts.Labels = labels;
    imgOpts.LabelMeanings = newsgroups;
    imgOpts.LabelSetNames = {'newsgroup'};
    imgOpts.DocFileNames = names';

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.9 0.8]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

 case '20NewsSubset2tfidf'

    load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub2train_TFIDFcorr_111809.mat');

    X = full(tdm_norm);

    % GUI data
    imgOpts.isTextData = true;
    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.Terms = terms';
    imgOpts.Labels = labels;
    imgOpts.LabelMeanings = newsgroups;
    imgOpts.LabelSetNames = {'newsgroup'};
    imgOpts.DocFileNames = names';

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.9 0.8]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

case '20NewsSubset3'

    load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub3combo_TFcorr_111909.mat');

    X = full(tdm_norm);

    % GUI data
    imgOpts.isTextData = true;
    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.Terms = terms';
    imgOpts.Labels = labels;
    imgOpts.LabelMeanings = newsgroups;
    imgOpts.LabelSetNames = {'newsgroup'};
    imgOpts.DocFileNames = names';

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.9 0.8]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

case '20NewsSubset4'

    load('/Users/emonson/Data/Fodava/EMoDocMatlabData/n20_sub4train_TFcorr_111809.mat');

    X = full(tdm_norm);

    % GUI data
    imgOpts.isTextData = true;
    imgOpts.hasLabels = true;
    imgOpts.hasLabelMeanings = true;
    imgOpts.hasLabelSetNames = true;
    imgOpts.hasDocFileNames = true;

    imgOpts.Terms = terms';
    imgOpts.Labels = labels;
    imgOpts.LabelMeanings = newsgroups;
    imgOpts.LabelSetNames = {'newsgroup'};
    imgOpts.DocFileNames = names';

    % EMonson options from older demo code
    GWTopts.knn = 50;
    GWTopts.knnAutotune = 10;
    GWTopts.smallestMetisNet = 5;

    % projection dimension
    GWTopts.AmbientDimension = inf; 

    GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

    GWTopts.errorType = 'relative';
    GWTopts.threshold0 = [0.9 0.8]; % threshold for choosing pca dimension at each nonleaf node
    GWTopts.precision  = 0.001; % only for leaf nodes

    GWTopts.coeffs_threshold = 0;
    GWTopts.sparsifying = false;

    % Default = true, which clears out the scaling functions at the
    % leaf nodes
    GWTopts.avoidLeafnodePhi = false;

  case 'NaturalImagePatches'

      load NaturalImagePatches.mat
      X = Y(:, randsample(size(Y,2),10000)); %#ok<NODEF>

      % image parameters
      imgOpts.imageData = true;
      imgOpts.imR = 16;
      imgOpts.imC = 16;
      imgOpts.isCompressed = false;

      % GWT parameters that need to be set separately
      %GWTopts.ManifoldDimension = 4; 
      GWTopts.errorType = 'relative';
      GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
      GWTopts.precision  = 0.05; % only for leaf nodes

  case 'IntegralOperator'

      [X0,~,~,Labels]=GenerateDataSets('Planes',struct('NumberOfPoints',5000,'EmbedDim',3,'PlanesOpts',struct('Number',2,'Dim',[1,2],'Origins',zeros(2,3),'NumberOfPoints',[1000,4000],'Dilations',[1,5])));
      X=GenerateIntegralOperatorFromSets(X0(:,Labels==1)', X0(:,Labels==2)');

      GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:

      GWTopts.errorType = 'relative';
      GWTopts.threshold0 = 0.1; % threshold for choosing pca dimension at each nonleaf node
      GWTopts.precision  = 0.01; % only for leaf nodes

  case 'MeyerStaircase'

      % data parameters
      dataset = struct();
      dataset.name = pExampleName;

      % % % %         dataset.N = 100; % number of data points to be generated            % This is for Cosine, gives Haar wavelets?
      % % % %         dataset.D = 5000; % ambient dimension of the data
      % % % %         dataset.k = 5000; % intrinsic dimension of the manifold

      % % % %         dataset.N = 40; % number of data points to be generated             % Another possibility for Cosine...
      % % % %         dataset.D = 8000; % ambient dimension of the data
      % % % %         dataset.k = 8000; % intrinsic dimension of the manifold

      dataset.N = 1000;
      dataset.k = 1;
      dataset.D = 1000;
      dataset.MeyerStepWidth=40;
      dataset.noiseLevel = 0/sqrt(dataset.D);

      % Generate data
      X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.D,'MeyerStepWidth',dataset.MeyerStepWidth,'EmbedDim',dataset.D,'NoiseType','Gaussian','NoiseParam',0) );

      % Add noise
      if dataset.noiseLevel>0,
          X = X_clean + dataset.noiseLevel*random('norm', 0,1, [dataset.N, dataset.D]); % N by D data matrix
      else
          X = X_clean;
      end

      % GWT parameters that need to be set separately
      GWTopts.X_clean = X_clean;
      GWTopts.ManifoldDimension = dataset.k;
      GWTopts.errorType = 'absolute';
      GWTopts.precision  = 5e-3; % only for leaf nodes     

  case 'D-Gaussian' % D-Gaussian
      %% data parameters
      dataset = struct();
      dataset.name = pExampleName;        
      dataset.N = 10000;
      dataset.k = 0;
      dataset.D = 512;
      dataset.noiseLevel = 0.01/sqrt(dataset.D);        

      % Generate data
      lFactor = 1/2;
      X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.D,'EmbedDim',dataset.D,'NoiseType','Gaussian','NoiseParam',0, ...
          'GaussianMean',[ones(5,1);lFactor^2*ones(10,1);lFactor^3*ones(20,1);lFactor^4*ones(40,1);zeros(dataset.D-75,1)]', ...
          'GaussianStdDev',0.2*[ones(5,1);lFactor^2*ones(10,1);lFactor^3*ones(20,1);lFactor^4*ones(40,1);zeros(dataset.D-75,1)]) ); % figure;plot(idct(X_clean(randi(size(X_clean,1),1),:)))

      % Add noise
      if dataset.noiseLevel>0,
          X = X_clean + dataset.noiseLevel*random('norm', 0,1, [dataset.D, dataset.N]); % N by D data matrix
      else
          X = X_clean;
      end

      %% GWT parameters that need to be set separately
      GWTopts.X_clean = X_clean;
      GWTopts.ManifoldDimension = 0;
      GWTopts.errorType = 'relative';
      GWTopts.threshold0 = lFactor/2; % threshold for choosing pca dimension at each nonleaf node
      GWTopts.precision  = lFactor/10; % only for leaf nodes        

  otherwise % artificial data
      %% data parameters
      dataset = struct();
      dataset.name = pExampleName;

      % % % %         dataset.N = 100; % number of data points to be generated            % This is for Cosine, gives Haar wavelets?
      % % % %         dataset.D = 5000; % ambient dimension of the data
      % % % %         dataset.k = 5000; % intrinsic dimension of the manifold

      % % % %         dataset.N = 40; % number of data points to be generated             % Another possibility for Cosine...
      % % % %         dataset.D = 8000; % ambient dimension of the data
      % % % %         dataset.k = 8000; % intrinsic dimension of the manifold

      dataset.N = 10000;
      dataset.k = 2;
      dataset.D = 50;
      dataset.noiseLevel = 0/sqrt(dataset.D);

      % Generate data
      X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.k,'EmbedDim',dataset.D,'NoiseType','Uniform','NoiseParam',0) );
      %X_clean = X_clean';         % This is for 'Cosine' only!!!!!

      % Add noise
      if dataset.noiseLevel>0,
          X = X_clean + dataset.noiseLevel*random('norm', 0,1, [dataset.N, dataset.D]); % N by D data matrix
          GWTopts.X_clean = X_clean;
      else
          X = X_clean;
      end
      %% GWT parameters that need to be set separately
      GWTopts.ManifoldDimension = dataset.k;
      %GWTopts.threshold0=0.5;
      GWTopts.errorType = 'absolute';
      GWTopts.precision  = 1e-3; % only for leaf nodes

end
fprintf('done. (%.3f sec)',toc);

% threshold for wavelet coefficients
GWTopts.coeffs_threshold = 0; %GWTopts.precision/10;

%figure; do_plot_data(X,[],struct('view', 'pca'));
