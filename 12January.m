m = prnist([0:9],[1:25:1000]);
data = seldat(m);
%resizing images so every image is same size
resized = im_resize(data,[128,128]);
dataset = im_features(resized);

%Leave out irrelevant features
dataset = dataset(:,[1 3:14 18:23]);

%Outlier removal
dataset = remoutl(dataset);

%Scaling data
%Maybe using the softmax scaling can be a good idea.
dataset = dataset*scalem(dataset,'variance');

%%
% Feature selection based on individual features
% Do values of the feature differ significantly for different classes?
% Following Paragraph 5.4 (with the modification of problem 5.4)
N = +classsizes(dataset);
[Mean Cov] = meancov(dataset);
Mean = +Mean;
Var = zeros(10,19);
for i = 1:10;
    Var(i,:) = diag(Cov(:,:,i));            %We scaled based on variance, total variance should be 1 right??
end

Sz = sqrt(sum(Var).^2./(sum(classsizes(dataset))-2));  


for j = 1:19
    for i = 1:10
        a = 1:10;
        a(a==i)=[];
        q(i,j) = Mean(i,j)-mean(Mean(a,j))/Sz(j);
    end
end

% q is the significance of mean value differences. Using max(abs(q)) the
% features with high discriminativity with respect to a certain class can
% be selected.

% Class 6,8,10,12,14 and 17 have high mean differences for some classes
plotf(dataset(:,14));


S_W = bsxfun(@sum,bsxfun(@times,Cov,reshape(N/sum(N),1,1,10)),3);
S_B = Mean'*bsxfun(@times,Mean,N'/sum(N));
S_M = cov(+dataset);
S_M2 = S_B + S_W;           %S_M2 and S_M don't perfectly match...


% Select features based on Scatter Matrices (par 5.6.3). Using a forward
% selection method. The best features are stored in the vector 'select'. 
% The value of their selection criterion in the vector 'J1'.
select = [];
for i = 1:5
    clear Performance
    for j = 1:19
        featloop = [select j];
        if any(select==j)
        else
            Performance(j) = trace(S_B(featloop',featloop))/trace(S_W(featloop',featloop));
        end
    end
    [J1(i),I] = max(Performance);
    select = [select I];
end
plotf(dataset(:,select));


%%
%classifing, knnc as example
features = dataset(:,select(1:2));
figure;
scatterd(features,'legend');
w = knnc(features,4);
hold on; plotc(w);
