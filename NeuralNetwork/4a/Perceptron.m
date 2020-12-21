function [ConfusionMatrix, Target, a] = Perceptron(dataset, eta, k)
[n,m] = size(dataset);
d = m-1;
%n = num of observations 
%m = d(=classes) + 1(=target)

%% if k = 2 ->Cross Validation: split the dataset into 2 parts (training and test) with equal size
if (k == 2)
    % split dataset into 2 parts (training and test) with equal size
    rand_indexes = randperm(n); 
    indexes.training = rand_indexes(1: n/2); 
    indexes.test = rand_indexes(n/2 + 1 : n);
    set.training = dataset(indexes.training, 1:d); 
    set.test = dataset(indexes.test, 1:d); 
    Target.training = dataset(indexes.training, m); 
    Target.test = dataset(indexes.test, m); 
    
    
    [error, w, a] = Perceptron_Predict(set.training, set.test, Target.training, Target.test, eta); 
    [ConfusionMatrix] = evaluation_classifier(error, a, Target.test);  




%% if k = n = num of observation -> perform leave-one-out cross validation  
elseif (k == n) 
    %split into n training sets ((n-1)x(d+1)) and n test sets (1 x d+1) in n possible ways 
    for i = 1:n 
        rand_indexes = randperm(n); 
        
        zero = find(rand_indexes == i);
        rand_indexes(zero) = [];
        
        indexes.test = i; 
        set.training = dataset(rand_indexes, 1:d); 
        set.test = dataset(indexes.test, 1:d); 
        Target.training = dataset(rand_indexes, m); 
        Target.test = dataset(indexes.test, m); 
        target(i,:) = dataset(indexes.test,m); 
        
        [error(:,:,i), w, a(:,:,i)] = Perceptron_Predict(set.training, set.test, Target.training, Target.test, eta); 
        [Single_ConfusionMatrix(:,:,i)] = evaluation_classifier(error(:,:,i), a(:,:,i), Target.test); 
        
        %Compute average ConfusionMatrices
        %ConfusionMatrix = zeros(2,2); 
        average1_1 = sum(Single_ConfusionMatrix(1,1,:)); 
        ConfusionMatrix(1,1) = average1_1/n; 
        average2_2 = sum(Single_ConfusionMatrix(2,2,:)); 
        ConfusionMatrix(2,2) = average2_2/n; 
        average1_2 = sum(Single_ConfusionMatrix(1,2,:)); 
        ConfusionMatrix(1,2) = average1_2/n; 
        average2_1 = sum(Single_ConfusionMatrix(2,1,:)); 
        ConfusionMatrix(2,1) = average2_1/n; 
        
            
    end

%% if k = k , 2<k<n -> perform k-fold cross validation 
elseif (2 < k < n) 
    %slit into k training sets with num.rows=n/k and corresponding test sets
    %with num.rows = n-n/k: 
    counter = 0;
    while (counter < k) 
        rand_indexes = randperm(n); 
        indexes.training = rand_indexes(1: n/k); 
        indexes.test = rand_indexes(n/k + 1 : n);
        set.training = dataset(indexes.training, 1:d); 
        set.test = dataset(indexes.test, 1:d); 
        Target.training = dataset(indexes.training, m); 
        Target.test = dataset(indexes.test, m); 
        for i = 1:k
            [error(:,:,k), w, a(:,:,i)] = Perceptron_Predict(set.training, set.test, Target.training, Target.test, eta);
            [Single_ConfusionMatrix(:,:,i)] = evaluation_classifier(error(:,:,i), a(:,:,i), Target.test);
        end 
        %Compute average ConfusionMatrices
        %ConfusionMatrix = zeros(2,2); 
        average1_1 = sum(Single_ConfusionMatrix(1,1,:)); 
        ConfusionMatrix(1,1) = average1_1/k; 
        average2_2 = sum(Single_ConfusionMatrix(2,2,:)); 
        ConfusionMatrix(2,2) = average2_2/k; 
        average1_2 = sum(Single_ConfusionMatrix(1,2,:)); 
        ConfusionMatrix(1,2) = average1_2/k; 
        average2_1 = sum(Single_ConfusionMatrix(2,1,:)); 
        ConfusionMatrix(2,1) = average2_1/k; 
        
        counter = counter + 1; 

    end 

%% if fold = k, with k<2 or k>n -> abort the run
else k < 2 || k > n 
    disp('error value k!'); 
end 

end

function [error, last_w, a] = Perceptron_Predict(training_set, test_set, training_target, test_target, eta) 
%% Fit the model with training set 
[n,d] = size(training_set); %m = num of classes
[m,b] = size(test_set); 
num_interactions = 10000; 

if d ~= b 
    disp('Error!') 
end 


w = [];    
    %Initialization of weight (random choice between -1 and 1)   
for l = 1:d %num of classes
    w(1,l) = 2.*rand(1,1) -1; 
end

for z = 1:num_interactions

    
    for j = 1:n %num of observation of training set
        
         %compute r as sum of products
         r = training_set(j,:) * w';

        %compute activation function : sign function is used 
        a = sign(r); 

        %check error 
        error = 0.5*(training_target(j) - a); 
        %compute delta and update w
        delta = (eta*error) * training_set(j,:); 
        w = delta + w; 
        delta = 0; 
        error = 0; 
    end 
end 
last_w = w; 


%% Classify Test set 
sum = 0 ; 
r = 0 ; 

%Compute the predicted target
for j = 1:m %num of observations of the test set 
    
        r = test_set(j,:) * (last_w)'; 
        
        a(j,:) = sign(r); 

    %Check error comparing real target test and predicted target test
    error(j,:) = 0.5*(test_target(j,:) - a(j,:));   
end 
end 

function [ConfusionMatrix] = evaluation_classifier(error, a, test_target)

%Compute Confusion Matrix 

TP = 0; 
TN = 0;

FP = 0; 
FN = 0; 
for j = 1:size(a,1)  
    if a(j,1) == 1 
        if test_target(j,1) == 1
            TP = TP + 1; 
        else 
            FP = FP + 1; 
        end 
    end
    
    if a(j,1) ==-1 
        if test_target(j,1) == -1
            TN = TN + 1; 
        else 
            FN = FN + 1; 
        end 
    end 
end 

ConfusionMatrix(1,1) = 100*(TP/size(a,1)); 
ConfusionMatrix(1,2) = 100*(FN/size(a,1)); 
ConfusionMatrix(2,1) = 100*(FP/size(a,1)); 
ConfusionMatrix(2,2) = 100*(TN/size(a,1)); 


    

    
        

end 

   
   

