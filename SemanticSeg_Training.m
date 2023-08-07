clear
close all
clc

% Image/Label load
outputFolder = fullfile('Path'); 
imgDir = fullfile(outputFolder,'images'); %Image
imds = imageDatastore(imgDir);

I = readimage(imds,50); % Example Image
I = histeq(I);
figure(1)
imshow(I)

classes = [
    "Background"
    "Patella"
    "Femur_1"
    "Tibia_1"
    "Tibia_2"
    "Tibia_3"
    "Tibia_4"
    "Tibia_5"
    "Tibia_6"
    "Tibia_7"
    "Tibia_8"
    "Tibia_9"
    "Tibia_10"
    "Tibia_11"
    "Tibia_12"
    "Tibia_13"
    "Tibia_14"
    "Tibia_15"
    "Tibia_16"
    "Tibia_17"
    "Tibia_18"
    "Tibia_19"
    "Tibia_20"
    "Tibia_21"
    "Tibia_22"
    "Tibia_23"
    "Fibula_1"
    "Fibula_2"
    "Fibula_3"
    "Fibula_4"
    "Fibula_5"
    "Fibula_6"
    "Fibula_7"
    "Fibula_8"
    "Fibula_9"
    "Fibula_10"
    "Fibula_11"
    "Fibula_12"
    ];

labelIDs = SegPixelLabelIDs();
labelDir = fullfile(outputFolder,'labels'); %Label
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

C = readimage(pxds,50); % Example Label
cmap = SegColorMap;
B = labeloverlay(I,C,'ColorMap',cmap,'Transparency',0.25);
figure(2)
imshow(B)
pixelLabelColorbar(cmap,classes);
%%
tbl = countEachLabel(pxds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure(3)
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
%%

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionSegData(imds,pxds);

numTrainingImages = numel(imdsTrain.Files) % Train (0.6)

numValImages = numel(imdsVal.Files) % Validation (0.2)

numTestingImages = numel(imdsTest.Files) % Test (0.2)

imageSize = [512 512 3];

% Specify the number of classes.
numClasses = numel(classes);

% Create DeepLab v3+ & Resnet 50.------------------------------------------
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;

imageFreq_L = length(imageFreq);

for i=1:imageFreq_L
    TF=isnan(imageFreq(i,1));
    if TF == 1
        imageFreq(i,1) = 0.000000001;
    end
end

classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

dsVal = combine(imdsVal,pxdsVal);
%%
% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-4, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',10, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', '', ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);

dsTrain = combine(imdsTrain, pxdsTrain);

% Data Augmentation
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));

doTraining = true; % Deep Learning strats
if doTraining    
    [net, info] = trainNetwork(dsTrain,lgraph,options);
end

figure(10)
plot(net) % Deep Learing Model Visualization

% Saving Weight File
Semantic_bone_Tibia_18 = net;
save Semantic_bone_Tibia_18detector


% Functions ---------------------------------------------------------------

function labelIDs = SegPixelLabelIDs()

labelIDs = {% "Background"
    [0 0 0]
    
    % "Patella"
    [207 136 128]

    % "Femur_1"
    [41 128 0]

    % "Tibia_1"
    [146 43 33]

    % "Tibia_2"
    [118 68 138]

    % "Tibia_3"
    [31 97 141]

    % "Tibia_4"
    [20 143 119]

    % "Tibia_5"
    [30 132 73]

    % "Tibia_6"
    [183 149 11]

    % "Tibia_7"
    [175 96 26]

    % "Tibia_8"
    [40 55 71]

    % "Tibia_9"
    [176 58 46]

    % "Tibia_10"
    [108 52 131]

    % "Tibia_11"
    [40 116 166]

    % "Tibia_12"
    [17 122 101]

    % "Tibia_13"
    [35 155 86]

    % "Tibia_14"
    [185 119 14]

    % "Tibia_15"
    [160 64 0]

    % "Tibia_16"
    [33 47 61]

    % "Tibia_17"
    [205 97 85]

    % "Tibia_18"
    [175 122 197]

    % "Tibia_19"
    [84 153 199]

    % "Tibia_20"
    [72 201 176]

    % "Tibia_21"
    [82 190 128]

    % "Tibia_22"
    [244 208 63]

    % "Tibia_23"
    [235 152 78]

    % "Fibula_1"
    [169 50 150]

    % "Fibula_2"
    [136 78 160]

    % "Fibula_3"
    [36 113 50]

    % "Fibula_4"
    [23 165 137]

    % "Fibula_5"
    [34 153 160]

    % "Fibula_6"
    [212 172 13]

    % "Fibula_7"
    [202 111 30]

    % "Fibula_8"
    [46 64 150]   

    % "Fibula_9"
    [203 67 53]

    % "Fibula_10"
    [125 60 152]

    % "Fibula_11"
    [46 134 193]

    % "Fibula_12"
    [19 141 30]};
end

function pixelLabelColorbar(cmap, classNames)

colormap(gca,cmap)

c = colorbar('peer', gca);

c.TickLabels = classNames;
numClasses = size(cmap,1);

c.Ticks = 1/(numClasses*2):1/numClasses:1;

c.TickLength = 0;
end

function cmap = SegColorMap()

cmap = [0 0 0
    207 136 128
    41 128 0
    146 43 33
    118 68 138
    31 97 141
    20 143 119
    30 132 73
    183 149 11
    175 96 26
    40 55 71
    176 58 46
    108 52 131
    40 116 166
    17 122 101
    35 155 86
    185 119 14
    160 64 0
    33 47 61
    205 97 85
    175 122 197
    84 153 199
    72 201 176
    82 190 128
    244 208 63
    235 152 78
    169 50 150
    136 78 160
    36 113 50
    23 165 137
    34 153 160
    212 172 13
    202 111 30
    46 64 150  
    203 67 53
    125 60 152
    46 134 193
    19 141 30];

% Normalize between [0 1].
cmap = cmap ./ 255;
end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionSegData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = SegPixelLabelIDs();

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

function data = augmentImageAndLabel(data, xTrans, yTrans)
% Augment images and pixel label images using random reflection and
% translation.

for i = 1:size(data,1)
    
    tform = randomAffine2d(...
        'XReflection',true,...
        'XTranslation', xTrans, ...
        'YTranslation', yTrans);
    
    % Center the view at the center of image in the output space while
    % allowing translation to move the output image out of view.
    rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');
    
    % Warp the image and pixel labels using the same transform.
    data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
    data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);
    
end
end
