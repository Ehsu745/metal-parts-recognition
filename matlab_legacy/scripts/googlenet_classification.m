clear; clc;


cd 'D:\matlab'
allData=imageDatastore('new','IncludeSubfolders',true,'LabelSource','foldernames');

load('googLeNet_4.mat')
[imds60,imds40]=splitEachLabel(allData,0.6,'randomized');
opts=trainingOptions('adam','InitialLearnRate',0.0001,'MaxEpochs',50,'MiniBatchSize',1,'Plots','training-progress',...
'ValidationData',imds40,'ExecutionEnvironment','multi-gpu');
myNet=trainNetwork(imds60,lgraph_1,opts);
desiredLabel=imds40.Labels;
predictedLabel=classify(myNet,imds40);
accuracy=mean(desiredLabel==predictedLabel)

confMat = confusionmat(desiredLabel, predictedLabel)
for i=1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(i,:)); % precision
end
for i=1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(:,i));% recall
end

F1 = (2 .* precision .* recall) / (precision + recall)% F1-score