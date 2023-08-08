%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 18th Aug 2023 by Do-Kun Yoon              %
%                  Industrial R&D Center, KAVILAB Co. Ltd                %
%                       Email: louis_youn@kavilab.ai                     %
%                              Co-developers                             %
%           Hyeonjoo Kim, Moo-Sub Kim, Juyeon You, Hayeong Cha           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reference
% 
% 1.Chen, L., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. 
% Encoder-Decoder with Atrous Separable Convolution for Semantic Image 
% Segmentation. In Proceedings of the European Conference on Computer 
% Vision (ECCV), 801-818 (2018).
% 
% 2.Chen, L., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. 
% DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, 
% Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on 
% Pattern Analysis and Machine Intelligence (TPAMI), 40(4), 834-848 (2018).
% 
% 3.Yang, Z., Li, W., Wang, X., & He, Y. Image segmentation algorithm with 
% adaptive attention mechanism based on Deeplab v3 plus. 
% Journal of Computer Applications, 42(1), 230 (2022).
% 
% 4.Yurtkulu, S. C., Şahin, Y. H., & Unal, G. Semantic segmentation with 
% extended DeepLabv3 architecture. 2019 27th Signal Processing and 
% Communications Applications Conference (SIU), 1-4 (2019).
% 
% 5.Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. Pyramid Scene 
% Parsing Network. Proceedings of the IEEE Conference on 
% Computer Vision and Pattern Recognition (CVPR), 6230-6239 (2017). 
% 
% 6.He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for 
% Image Recognition. Proceedings of the IEEE Conference on 
% Computer Vision and Pattern Recognition (CVPR), 770-778 (2016). 
% 
% 7.Wu, Z., Shen, C., & Hengel, A. V. D. Wider or Deeper: Revisiting 
% the ResNet Model for Visual Recognition. Pattern Recognition, 
% 90, 119-133 (2019). 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
clc


% Image/Label load
outputFolder = fullfile('Data Path'); % Please write path for dataset 
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

tbl = countEachLabel(pxds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure(3)
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionSegData(imds,pxds); %Function

numTrainingImages = numel(imdsTrain.Files) % Train (0.6)

numValImages = numel(imdsVal.Files) % Validation (0.2)

numTestingImages = numel(imdsTest.Files) % Test (0.2)

imageSize = [512 512 3];

% Specify the number of classes.
numClasses = numel(classes);


% ----------------------------Network Model -------------------------------
lgraph = layerGraph();

tempLayers = imageInputLayer([512 512 3],"Name","input_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","aspp_Conv_1","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_1")
    reluLayer("Name","aspp_Relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_2","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_2")
    reluLayer("Name","aspp_Relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_3","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_3")
    reluLayer("Name","aspp_Relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_4","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_4")
    reluLayer("Name","aspp_Relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","catAspp")
    convolution2dLayer([1 1],256,"Name","dec_c1","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn1")
    reluLayer("Name","dec_relu1")
    transposedConv2dLayer([8 8],256,"Name","dec_upsample1","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","dec_c2","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn2")
    reluLayer("Name","dec_relu2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","dec_crop1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","dec_cat1")
    convolution2dLayer([3 3],256,"Name","dec_c3","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn3")
    reluLayer("Name","dec_relu3")
    convolution2dLayer([3 3],256,"Name","dec_c4","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn4")
    reluLayer("Name","dec_relu4")
    convolution2dLayer([1 1],38,"Name","scorer","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    transposedConv2dLayer([8 8],38,"Name","dec_upsample2","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    crop2dLayer("centercrop","Name","dec_crop2")
    softmaxLayer("Name","softmax-out")
    pixelClassificationLayer("Name","labels")];
lgraph = addLayers(lgraph,tempLayers);

clear tempLayers;

lgraph = connectLayers(lgraph,"input_1","conv1");
lgraph = connectLayers(lgraph,"input_1","dec_crop2/ref");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"activation_10_relu","dec_c2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_1");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_2");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_3");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_4");
lgraph = connectLayers(lgraph,"aspp_Relu_1","catAspp/in1");
lgraph = connectLayers(lgraph,"aspp_Relu_2","catAspp/in2");
lgraph = connectLayers(lgraph,"aspp_Relu_3","catAspp/in3");
lgraph = connectLayers(lgraph,"aspp_Relu_4","catAspp/in4");
lgraph = connectLayers(lgraph,"dec_upsample1","dec_crop1/in");
lgraph = connectLayers(lgraph,"dec_relu2","dec_crop1/ref");
lgraph = connectLayers(lgraph,"dec_relu2","dec_cat1/in1");
lgraph = connectLayers(lgraph,"dec_crop1","dec_cat1/in2");
lgraph = connectLayers(lgraph,"dec_upsample2","dec_crop2/in");

% -------------------------------------------------------------------------

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
lgraph = replaceLayer(lgraph,"labels",pxLayer);

dsVal = combine(imdsVal,pxdsVal);

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
plot(net)


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
    
    cmap = cmap ./ 255;
end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionSegData(imds,pxds)
    rng(0); 
    numFiles = numel(imds.Files);
    shuffledIndices = randperm(numFiles);
    
    numTrain = round(0.60 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);
    
    numVal = round(0.20 * numFiles);
    valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
    
    testIdx = shuffledIndices(numTrain+numVal+1:end);
    
    trainingImages = imds.Files(trainingIdx);
    valImages = imds.Files(valIdx);
    testImages = imds.Files(testIdx);
    
    imdsTrain = imageDatastore(trainingImages);
    imdsVal = imageDatastore(valImages);
    imdsTest = imageDatastore(testImages);
    
    classes = pxds.ClassNames;
    labelIDs = SegPixelLabelIDs();
    
    trainingLabels = pxds.Files(trainingIdx);
    valLabels = pxds.Files(valIdx);
    testLabels = pxds.Files(testIdx);
    
    pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
    pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
    pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

function data = augmentImageAndLabel(data, xTrans, yTrans)
    for i = 1:size(data,1)
        
        tform = randomAffine2d(...
            'XReflection',true,...
            'XTranslation', xTrans, ...
            'YTranslation', yTrans);

        rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');

        data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
        data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);
        
    end
end
