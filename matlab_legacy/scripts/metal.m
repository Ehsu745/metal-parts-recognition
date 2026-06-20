% clc; clear; close all;

cd     'C:\school\prog\matlab\toy'
imgTrain = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
imgTest = imageDatastore('test','IncludeSubfolders',true,'LabelSource','foldernames');
train_data = load('train\train_label.mat');
trainingData = objectDetectorTrainingData(train_data.gTruth);
layers = 'alexnet';
options = trainingOptions('adam','MaxEpochs',1,'MiniBatchSize',75,'initialLearnRate',0.0001,'Shuffle','every-epoch','CheckpointPath',tempdir);
detector = trainFasterRCNNObjectDetector(trainingData, layers, options);
test_data = load('test\test_label.mat');
testingData = objectDetectorTrainingData(test_data.gTruth);
overlapRatio= [];
for i = 1:size(testingData,1)
    img = imread(testingData.imageFilename{i});
    [bboxes,scores,labels] = detect(detector,img);
    
    bbox_g = zeros(6,4);
    if isnan(testingData.Bolt{i})
        testingData.Bolt{i} = [0, 0, 0, 0];
    end
    if isnan(testingData.Foundationbolt{i})
        testingData.Foundationbolt{i} = [0, 0, 0, 0];
    end
    if isnan(testingData.Gasket{i})
        testingData.Gasket{i} = [0, 0, 0, 0];
    end
    if isnan(testingData.drillingscrew{i})
        testingData.drillingscrew{i} = [0, 0, 0, 0];
    end
    if isnan(testingData.nut{i})
        testingData.nut{i} = [0, 0, 0, 0];
    end
    if isnan(testingData.screw{i})
        testingData.screw{i} = [0, 0, 0, 0];
    end
    bbox_g = [testingData.Bolt{i};testingData.Foundationbolt{i}; testingData.Gasket{i};testingData.drillingscrew{i};testingData.nut{i};testingData.screw{i}];
    %需要辨識6種類別
    
    labels_g = zeros(8,1);
    for i = 1:size(testingData.Label)
        labels_g = testingData.Label{i}
    end

    if~isempty(bboxes)
        temp_overlapRatio = bboxOverlapRatio(bboxes,bbox_g);
        %需要把overlap改成 1*6大小
        overlapRatio(i) = mean(temp_overlapRatio(:));
        l = insertObjectAnnotation(img, "rectangle", bboxes, labels);
        figure, imshow(l)
    else
        overlapRatio(i) = 0;
        figure,imshow(img)
    end

    %比對程式
    
end