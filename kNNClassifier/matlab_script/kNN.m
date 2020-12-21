function [classification, error_rate] = kNN(training_set, test_set, k, test_class_label)
%% Task2_1 : Check that the number of arguments received (nargin) equals at least the number of mandatory arguments
if nargin <= 2 
    disp('Error:the number of arguments received must be equals at least the number of mandatory arguments'); 
end
%% Task 2_2 : Check that the number of columns of the second matrix equals the number of columns of the first matrix
[n, d] = size(training_set); 
[m, c] = size(test_set); 

if d ~= c+1  
    disp('Error, wrong Matrices dimensions'); 
end

%% Task 2_3 : Check that k>0 and k<=cardinality of the training set (number of rows, above referred to as n)

if (k <= 0 || k > d) 
    disp('Error, wrong k'); 
end

%% Task 2_4 : Classify the test set

% compute the euclidian distance
[D, I] = pdist2(training_set(:,1:c), test_set, 'euclidean', 'Smallest', k); 
% I = K by MY matrix containing indices of the observations in X corresponding to the K smallest pairwise distances in D. 

class = zeros(k, m); 
for i = 1:k 
    for j = 1:m
        class (i,j) = training_set(I(i,j), end); 
    end 
end 
%Compute classification 
classification = zeros(m, 1); 
for i=1:m 
    classification(i,1) = mode(class(:,i)); 
end
%% Task 2_5 : Compute error_rate if we have also the test_class_labels for comparison 
if nargin > 3 
    error_rate = (sum(classification ~= test_class_label))/m;
end 

   


end

