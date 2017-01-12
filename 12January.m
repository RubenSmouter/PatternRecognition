m = prnist([0:9],[1:25:1000]);

data = seldat(m);
%resizing images so every image is same size
resized = im_resize(data,[128,128]);
dataset = im_features(resized);
dataset = dataset(:,[1 3:14 18:23]);
%Scaling data
dataset = dataset*scalem(dataset,'variance');

[testdata,trainingdata] = gendat(dataset,.5);



for f1 = 1:19
selectfeatures = dataset(:,[10 17 11 12 9 f1]);
[U,G] = meancov(selectfeatures);
S_W = bsxfun(@sum,G/10,3);
M = cell2mat(dset2cell(U));
S_B = M(:,:)'*M(:,:)/10; % Formula from 5.6.3, at the scaling the mean of M is already shifted to the origin
Performance(f1) = trace(S_B/S_W);
end


E = cleval(trainingdata(:,[9 10 11 12 17]),knnc);
plote(E);

%classifing, knnc as example
for i = 1:4
features = selectfeatures(:,[5 i]);
figure;
scatterd(features,'legend');
w = knnc(features,4);
hold on; plotc(w);
end
