clear all;
close all;
clc;

[XTrain,YTrain] = digitTrain4DArrayData;
size(XTrain)      %images
size(YTrain)      %correct answer labels

XTrain=1-XTrain;  % Reverse the black and white colors.  Save and run the program to see the difference. 

perm = randperm(size(XTrain,4),20);  % Randomize the order of images in XTrain 
for i = 1:20
subplot(4,5,i);
imshow(XTrain(:,:,:,perm(i)));
end
 
layers = [
imageInputLayer([28 28 1])   
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
averagePooling2dLayer(7)
 
fullyConnectedLayer(10)     % 10 output layer nodes
softmaxLayer
classificationLayer];  %close the bracket

options = trainingOptions('sgdm', ...
'InitialLearnRate',0.1, ...
'MaxEpochs',20, ...
'Verbose',false, ...
'Plots','training-progress', ...
'Shuffle','every-epoch' ); 

net1 = trainNetwork(XTrain,YTrain,layers,options);
save digitnet net1;


camera = webcam; % Connect to the camera

figure;  % open new figure window
while true    %this is a loop that will go on forever unless you break out of it.
im = snapshot(camera); % Take a picture
image(im); % Show the picture
im=rgb2gray(im); %make the image grayscale (that's what the network is expecting)
im=round(double(im)/255);  % change from 0-255 integer to 0-1 double (increase contrast by using round)
im = imresize(im,[28 28]); % Resize the picture for the network you trained (it's expecting a 28x28 image)
label = classify(net1,im); % Classify the picture.  Type help classify in the command window to get more information about this command.
title(char(label),'fontsize',18); % Show the class label
drawnow   %force matlab to immediately display the image and label
end