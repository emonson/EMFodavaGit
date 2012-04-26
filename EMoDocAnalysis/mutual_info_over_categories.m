% Mutual information of terms over document categories
% Formula gathered from this page:
% http://en.wikipedia.org/wiki/Cluster_labeling

% Trying this out since I was a bit worried that the "magic" science news
% data set had terms which were picked through "cheating" by looking at
% mutual information over the (known) category labels...

data_dir = '/Users/emonson/X_Archives/EMoDocAnalysis/X_Data/SelfTokenizedTDMs';
data_name = 'tdm_emo1nvj_112509.mat';
cd(data_dir);
load(data_name);

labels = dlmread('pure.classes');
classes = labels(:,[2 1]);

% Get rid of any "non-pure" classes (assigned 0 in X20)
%   by using filename integer converted to array index
%   Note: this takes care of missing 10976 file...
tdm = tdm(:,classes(:,2)-10000+1);

pC1tmp = zeros(1,8);
pC0tmp = zeros(1,8);
for ii = 1:8,
    pC1tmp(ii) = sum(classes(:,1)==ii)/length(classes(:,1));
    pC0tmp(ii) = sum(classes(:,1)~=ii)/length(classes(:,1));
end

pC1 = repmat(pC1tmp,[size(tdm,1) 1]);
pC0 = repmat(pC0tmp,[size(tdm,1) 1]);

pT1tmp = sum(tdm>0,2)/size(tdm,2);
pT0tmp = sum(tdm==0,2)/size(tdm,2);

pT1 = repmat(pT1tmp,[1 8]);
pT0 = repmat(pT0tmp,[1 8]);

pC1T1 = zeros(size(tdm,1),8);
pC1T0 = zeros(size(tdm,1),8);
pC0T1 = zeros(size(tdm,1),8);
pC0T0 = zeros(size(tdm,1),8);
numDocs = length(classes(:,1));

for ii = 1:8,
    c1 = repmat((classes(:,1)==ii)',[size(tdm,1) 1]);
    c0 = repmat((classes(:,1)~=ii)',[size(tdm,1) 1]);
    pC1T1(:,ii) = sum(c1 & (tdm>0),2)/numDocs;
    pC1T0(:,ii) = sum(c1 & (tdm==0),2)/numDocs;
    pC0T1(:,ii) = sum(c0 & (tdm>0),2)/numDocs;
    pC0T0(:,ii) = sum(c0 & (tdm==0),2)/numDocs;
end

I11 = pC1T1.*log2(pC1T1./(pC1.*pT1));
I10 = pC1T0.*log2(pC1T0./(pC1.*pT0));
I01 = pC0T1.*log2(pC0T1./(pC0.*pT1));
I00 = pC0T0.*log2(pC0T0./(pC0.*pT0));

MI = I11 + I10 + I01 + I00;
MI(isnan(MI)) = 0;
MImax = max(MI,[],2);

% Throwing out terms based on low mutual information over categories
tdm3 = tdm;
terms3 = terms;

throw_out = MImax<0.0075;
tdm3(throw_out,:) = [];
terms3(throw_out) = [];

% Plot of number of terms per document
figure; plot(sum(tdm3>0,1),'k.');

% Calculating "conditional" entropy
sum3 = sum(tdm3,2);
numel3 = sum(tdm3>0,2);
ratio3 = tdm3./repmat(sum3,[1 size(tdm3,2)]);
lratio3 = log(ratio3);
lratio3(isinf(lratio3)) = 0;
h3 = -1*sum(ratio3.*lratio3,2); %./log(numel3);
% h3(isnan(h3)) = 0;
figure; hist(h3,100);

% Calculating "normal" entropy
sum3b = sum(tdm3,2);
numel3b = sum(tdm3>0,2);
ratio3b = tdm3./repmat(sum3b,[1 size(tdm3,2)]);
lratio3b = log(ratio3b);
lratio3b(isinf(lratio3b)) = 0;
h3b = -1*sum(ratio3b.*lratio3b,2);
figure; hist(h3b,100);
