%% Lab 5b - practice with deep learning
% Romano Sara - S 4802844
%% Task 1 - Use pretrained Convolutional Neural Networks (convnet or CNN) 

%use the pretrained networks as black boxed to classify new images. 

%Load Pretrained Network
net = googlenet;
inputSize = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)))

%Read and Resize Image
I = imread('Tazza2.jpg'); 
%figure 
%imshow(I)

size(I) 

%Resize the image to the input size of the network 
I = imresize(I, inputSize(1:2)); 


%Classify Image 
[label, scores] = classify(net,I); 
label 

figure
imshow(I) 
title(string(label) + "," + num2str(100*scores(classNames == label),3) + "%"); 

%Display Top Predictions 
[~,idx] = sort(scores,'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)