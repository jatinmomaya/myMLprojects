clear all;
close all;
clc;

mydata = imageDatastore('MerchData',...
    'includeSubfolders',true,...
    'LabelSource','foldernames');

[mydataTrain, mydataValidation] =...
    splitEachLabel(mydata,0.7);

mydataResizedTrain = augmentedImageDatastore([224 224], mydataTrain,'colorpreprocessing','gray2rgb');
mydataResizedValidation = augmentedImageDatastore([224 224], mydataValidation,'colorpreprocessing','gray2rgb');

load MerchNewLayers;

options = trainingOptions('sgdm',...
    'MiniBatchSize',10,...
    'MaxEpochs',6,...
    'InitialLearnRate',1e-4,...
    'Shuffle','every-epoch',...
    'ValidationData',mydataResizedValidation,...
    'ValidationFrequency',6,...
    'Verbose', false,...
    'Plots','training-progress');

net = trainNetwork(mydataResizedTrain,lgraph_1,options);
[YPred,probs] = classify(net,mydataResizedValidation);
accuracy = mean(YPred == mydataValidation.Labels)

figure;
for i = 1:4
    subplot(2,2,i)
    a = randi(20);
    I = readimage(mydataValidation,a);
    imshow(I)
    label = YPred(a);
    title(string(label) + "," + num2str(100*max(probs(a,:)),3) + "%");
end

