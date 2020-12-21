%% Lab 4a - Single-Unit Neural Networks 
%Romano Sara s4802844
clear all; 
clc; 
close all;

%% Task 2: Load Dataset 
%1)Load iris dataset%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
iris_dataset=load('iris-2class.txt'); 

%2)Load MNIST dataset%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Training.set, Training.class_labels] = loadMNIST(0); 
[Test.set, Test.class_labels] = loadMNIST(1); 


%couple of digits: 5, 8: 
five = find(Training.class_labels == 5); 
eight = find(Training.class_labels == 8); 

Training_five = Training.set(five,:); 
Training_eight = Training.set(eight, :); 
Label_five = Training.class_labels(five,:); 
Label_eight = Training.class_labels(eight,:); 
MNIST.training = [Training_five; Training_eight]; 
MNIST.label = [Label_five; Label_eight]; 

MNIST_dataset = [MNIST.training, MNIST.label]; 

%set label 1 for 5 and -1 for 8 
for i = 1: size(MNIST_dataset,1)
    if MNIST_dataset(i,end) == 5
        MNIST_dataset(i,end) = 1; 
    elseif MNIST_dataset(i,end) == 8
        MNIST_dataset(i,end) = -1;
    end 
end

%order randomly the indexes 
rand_indexes = randperm(size(MNIST_dataset,1));
MNIST_dataset = MNIST_dataset(rand_indexes(1:500),:); 



%% Perform Tests in different conditions 
%%%%%%% case 1: Tests on IRIS dataset %%%%%%%%%%%%

k = [2,30,150]; 
% for i = 1:3
%     %eta = 0.1 
%     [ConfusionMatrix_iris_Perceptron, ~, ~] = Perceptron(iris_dataset,0.1,k(i));
%     showTable('iris' , 'perceptron', ConfusionMatrix_iris_Perceptron, k(i), 0.1)
%     
%     [ConfusionMatrix_iris_Adaline,  ~, ~] = Adaline(iris_dataset, 0.1, k(i)); 
%     showTable('iris' , 'adaline', ConfusionMatrix_iris_Adaline, k(i), 0.1) 
%     
%     %eta = 0.5 
%     [ConfusionMatrix_iris_Perceptron, ~, ~] = Perceptron(iris_dataset,0.5,k(i));
%     showTable('iris' , 'perceptron', ConfusionMatrix_iris_Perceptron, k(i), 0.5)
%     
%     [ConfusionMatrix_iris_Adaline,  ~, ~] = Adaline(iris_dataset, 0.5, k(i)); 
%     showTable('iris' , 'adaline', ConfusionMatrix_iris_Adaline, k(i), 0.5) 
%     
% end 

%%%%%%% case 1: Tests on MNIST dataset %%%%%%%%%%%%

for i = 1:3
    %eta = 0.1 
    [ConfusionMatrix_MNIST_Perceptron, ~, ~] = Perceptron(MNIST_dataset,0.1,k(i));
    showTable('MNIST' , 'perceptron', ConfusionMatrix_MNIST_Perceptron, k(i), 0.1)
    
    [ConfusionMatrix_MNIST_Adaline,  ~, ~] = Adaline(MNIST_dataset, 0.1, k(i)); 
    showTable('MNIST' , 'adaline', ConfusionMatrix_MNIST_Adaline, k(i), 0.1) 
    
    %eta = 0.5 
    [ConfusionMatrix_MNIST_Perceptron, ~, ~] = Perceptron(MNIST_dataset,0.5,k(i));
    showTable('MNIST' , 'perceptron', ConfusionMatrix_MNIST_Perceptron, k(i), 0.5)
    
    [ConfusionMatrix_MNIST_Adaline,  ~, ~] = Adaline(MNIST_dataset, 0.5, k(i)); 
    showTable('MNIST' , 'adaline', ConfusionMatrix_MNIST_Adaline, k(i), 0.5) 
    
end 

        
        
%% Table for Confusion Matrix
function showTable(dataset, algorithm, ConfusionMatrix, k, eta)
    TT = table(ConfusionMatrix(:,1),ConfusionMatrix(:,2));
    TT.Properties.VariableNames = {'Positive Predicted Target' 'Negative Predicted Target'};
    TT.Properties.RowNames = {'Positive Real Target' 'Negative Real Target'};

     f = figure; 
     uitable( 'Data',TT{:,:},'ColumnName',TT.Properties.VariableNames,...
        'RowName',TT.Properties.RowNames,  'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

    figName=sprintf('ConfusionMatrix_%s_%s_with_k=%d_and_eta=%d.jpg',dataset, algorithm, k, eta);
    saveas(f, figName);
end 

%% plot 
% 
% uno = find(dataset(:,3) == 1); 
% dataset1_1 = dataset(uno, 1:3); 
% zero = find(dataset(:,3) == (-1)); 
% dataset1_0 = dataset(zero, 1:3); 
% t = [1,150]; 
% y1 = w(end,1) * dataset(:,1); 
% y2 = w(end,2) * dataset(:,2); 
% figure; 
% hold on; 
% plot(dataset1_1(:,1), dataset1_1(:,2),'bx');
% plot(dataset1_0(:,1), dataset1_0(:,2),'rx');
% plot(dataset(:,1), y1); 
% plot(y2, dataset(:,2)); 
% 


