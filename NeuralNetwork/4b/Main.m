%% Lab 4b : Neural Networks 
%Romano Sara -- s4802844
close all; 
clear all; 
clc; 

%% Task 0 : Neural networks in Matlab 
%Execution of Matlab tutorial: FIT DATA WITH A NEURAL NETWORK 

%load the dataset 
%the chosen one is the chemical dataset 
load chemical_dataset; 
%NOTA : this dataset is one of the sample data sets that is part of the
%toolbox...
%thus, the structure of the dataset has rows and columns ordered according to
%the matlab neural network function generated. 

%to have more infos about the dataset, use "help chemical_dataset" on the
%command window.


%define the num of the hidden layer 
hiddenLayerSize = 10;


FittingProblemwithNeuralNetwork(chemicalInputs,chemicalTargets, hiddenLayerSize);

%% Task 1 : Feedforward multi-layer networks (multi-layer perceptrons)
%Execution of Matlab tutorial: CLASSIFY PATTERNS WITH A NEURAL NETWORK

%load the dataset
%the chosen one is the wine dataset 
load wine_dataset; 
%NOTA : this dataset is one of the sample data sets that is part of the
%toolbox...
%thus, the structure of the dataset has rows and columns ordered according to
%the matlab neural network function generated.

%to have more infos about the properties of the dataset, write on the
%command window "help wine_dataset" 

%load another dataset: glass dataset 
load glass_dataset;

PatternRecognitionProblemwithNeuralNetwork(wineInputs, wineTargets, hiddenLayerSize); 
PatternRecognitionProblemwithNeuralNetwork(glassInputs, glassTargets, hiddenLayerSize); 
