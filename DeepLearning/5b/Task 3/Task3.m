%% Lab 5b - practice with deep learning
% Romano Sara - S 4802844
clear all; 
close all; 
clc; 

%% Task 3: Transfer learning via retraining
%how to use an existing network as the initialization for a new training

%Pretrained image classigication networks have been trained on onver a
%milion images and can classify images into 1000 object categories. 

%The network takes an image as input and then outputs a label for the
%object in the image together with the probabilities for each of the object
%categories. 


%Load data 
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

%Load Pretrained Network 
net = googlenet

%Use 'analizyNetwork' function to display an interactive visualization of
%the network architecture and detailed information about the network
%layers. 

analyzeNetwork(net)

%Resize the image 
net.Layers(1)
inputSize = net.Layers(1).InputSize; 

%Replace Final Layers 

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 



numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%Classification layer specifies the output classes of the network 
% Replace the classification layer with a new one without class labels
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%To check that the new layers are connected correctly, plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%The last layer must be : "new_classoutput" 


%Freeze Initial Layers 
%The network is now ready to be retrained on the new set of images
%Optionally, you can "freeze" the weights of earlier layers in the network by setting the learning rates in those layers to zero. 
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%Train Network
%The network requires input images of size 224-by-224-by-3, but the images in the image datastore have different sizes

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%To automatically resize the validation images without performing further data augmentation, use an augmented image datastore without specifying any additional preprocessing operations.

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%Specify the number of epochs to train for
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Train the network using the training data.
net = trainNetwork(augimdsTrain,lgraph,options);

%Classify Validation Images 
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%Display four sample validation images with predicted labels 
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end


