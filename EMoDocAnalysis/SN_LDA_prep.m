load('X20.mat');

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

%% Write file for LDA-c (Blei)

ff = fopen('sn.dat', 'w');

for ii = 1:size(tdm,2),
    if ~mod(ii, 50), 
        fprintf(1, '%d / %d\n', ii, size(tdm,2)); 
    end;
    
    [rr,cc,vv] = find(tdm(:,ii));
    fprintf(ff, '%d', sum(vv > 0));
    
    for jj = 1:length(vv),
        % c-style 0-based term indices
        fprintf(ff, ' %d:%d', [rr(jj)-1 vv(jj)]);
    end;
    
    fprintf(ff, '\n');
end;

fclose(ff);

%% Write vocab file

ff = fopen('sn_vocab.txt', 'w');

for ii = 1:length(dict),
    ss = dict{ii};
    fprintf(ff, '%s\n', ss);
end

fclose(ff);

%%
% calculate TFIDF (std) normalization for word counts
% nkj = sum(tdm,1)';      % how many terms in each document
% D = size(tdm,2);        % number of documents
% df = sum(tdm>0,2);      % number of documents each term shows up in
% idf = log(D./(1+df));   % the 1+ is common to avoid divide-by-zero
% 
% [ii,jj,vv] = find(tdm);
% vv_norm = (vv./nkj(jj)).*idf(ii);
% 
% tdm_norm = sparse(ii,jj,vv_norm);
% X = full(tdm_norm);

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

