%% Lab Assignment 3 - kNN Classifier 
%Romano Sara 4802844    

clear all; 
clc;
close all; 

%% Task 1: Obtain a data set
%use loadMNIST.m function to obtain data set 

[Training.set, Training.class_labels] = loadMNIST(0); 
[Test.set, Test.class_labels] = loadMNIST(1); 

k = [1,2,3,4,5,10,15,20,30,40,50];
dim_k = max(size(k)); 

%% Task 2: Build a kNN classifier

for i = 1:dim_k
    
    [classification, error_rate(i, 1)] = kNN([Training.set(1:6000,:) Training.class_labels(1:6000,:)], Test.set(1:1000,:), k(i), Test.class_labels(1:1000,:));

end

error_rate_k = [k' error_rate];

%plot bar graph to check the error rate 
figure 
bar(error_rate_k(:,1), error_rate_k(:,2), 'BarWidth', 1);
xlabel('k'); 
ylabel('error'); 
title('Error rate in task 2'); 

%% Task 3 : Test the kNN classifier
%Compute the accurancy on the test set on 10 tasks: each digit vs the
%remaining 9, and for several values of k 

 kappa = [1,2,3,4,5,10,15,20,40,50, 100]; %new k values 
 
 d = [1,2,3,4,5,6,7,8,9,10]; %digits
 

 
 for j =1:max(size(d)) %Computing the classification and the error for each digit = [1:1:10]
 %in this case, considering just a single digit, the classification can be
 %just 0 or 1. 
     for i = 1: max(size(kappa)) 
        [classification2, error_rate_kap(i, 1)] = kNN([Training.set(1:6000,:), Training.class_labels(1:6000,:) == j], Test.set(1:1000,:), kappa(i), Test.class_labels(1:1000,:) == j);
        
     end
     %Fill in a matrix the error of each digit 
     classification_2(:,j) = classification2;
     error_rate_kappa(:,j) = error_rate_kap; 
    
 end
% %% Plotting error rates in bar graph
% for j = 1:10
%     figure 
%     bar(kappa, error_rate_kappa(:,j), 'BarWidth', 1);
%     xlabel('kappa'); 
%     ylabel('error rate of digit' + string(j));
% end
%   
%% Compute and plot accurancy 

%accuracy = (1-error_rate)*100 

accuracy = zeros(size(error_rate_kappa)); 
for i = 1:max(size(kappa)) 
    for j = 1:max(size(d))
        accuracy(i,j) = (1 - error_rate_kappa(i,j))*100; 
    end
end 

%%%%%%%%%%%PLOT AVERAGE ACCURACY WRT KAPPA VALUES %%%%%%%%%%%%%
%compute mean of accuracy
mean_accuracy = zeros(max(size(kappa)),1); 
for i = 1:max(size(kappa)) %for each k 
    mean_accuracy(i,1) = sum(accuracy(i,:)); 
end 

mean_accuracy = mean_accuracy/max(size(kappa)); 
    
%plot the average of the accuracy with respect to k 
figure 
plot(kappa, mean_accuracy); 
xlabel('k values');
ylabel('average accuracy'); 
title('Average accuracy with respect to k'); 


%%%%%%%%%%%%PLOT THE ACCURACY OF EACH DIGIT WITH RESPECT TO K %%%%%%%%%%%%%
figure; hold on;
for i = 1:size(accuracy, 2)
    plot(kappa, accuracy(:,i)); 
    xlabel('kappa values'); 
    ylabel('accuracy'); 
    legend('Digit 1','Digit 2','Digit 3', 'Digit 4', 'Digit 5', 'Digit 6','Digit 7','Digit 8','Digit 9','Digit 10' ); 
end 


%%%%%%%%%% ACCURACY BAR GRAPF OF EACH DIGIT WITH RESPECT TO K %%%%%%%%%%%%%

for i = 1: size(accuracy,2) 
    figure; 
    bar(kappa, accuracy(:,i), 'BarWidth', 1); 
    xlabel('kappa value'); 
    ylabel('accuracy digit ' + string(i)); 
end 

