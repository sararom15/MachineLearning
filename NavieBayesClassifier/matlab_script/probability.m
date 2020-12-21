function [B, num, Target, P] = probability(training_set)
[num.rows, num.columns] = size(training_set); 
%% Compute prior probability of class: P(c)
Target= training_set(:,num.columns);
%Calculate num of levels of the target attribute
A = unique(Target); 
num.levels_classes = size(A,1); %num of levels of target attribute %2

%Compute how many level i is present in the training step
num.appearance_of_each_level_class = zeros(1,num.levels_classes);
for j = 1:num.levels_classes
    
    for i = (Target)'
            if (i == j)
                num.appearance_of_each_level_class(j) = num.appearance_of_each_level_class(j) + 1; 
            end 
    end
end
%num.ofeach_levels_in_the_attribute_class %num of each level i for the training step 
P.prior_probability_class = zeros(1,num.levels_classes); 
for j=1:num.levels_classes
    P.prior_probability_class(j) = num.appearance_of_each_level_class(j)/num.rows; 
end
%P.prior_probability_class


%% Compute Likelihoods P(x|c) 
%num of columns - 1 is the num of predictors
num.predictors = num.columns - 1 ; 

%find the num of levels for each predictor i

%we have to find a vector.rows with all possible level for each
%predictors.
num.levels_for_each_input = zeros(1, num.predictors); 
for i = 1:num.predictors
    b = training_set(:,i); 
    a = unique(b); 
    num.levels_for_each_input(i) = length(a);    
end
%num.levels_for_each_input %vector.row with all num of levels of each predictors 

num.max_levels = max(num.levels_for_each_input); %3

%inizializiation of a null 3D matrix
B = zeros(num.predictors, num.max_levels, num.levels_classes); 

%compute how many class levels  i (of target attribute) areassociated with a
%possible level s of the predictor j 


for e = 1:num.rows %taking the num of instances
    i = Target(e); %can be 1 or 2 (yes or no) 
    for j = 1:num.predictors 
        for s = 1:num.levels_for_each_input(j)
            if training_set(e,j) == s
                B(j,s,i) = B(j,s,i) + 1; 
            end
        end
    end
end
%B


P.likelihoods = zeros(num.predictors, num.max_levels, num.levels_classes);
for i = 1:num.levels_classes
    for j = 1:num.predictors 
        for s=1:num.levels_for_each_input(j)
            P.likelihoods(j,s,i) = B(j,s,i)/num.appearance_of_each_level_class(i); 
        end
    end
end
%P.likelihoods 
            


end




         
    
  
       


    
    


            
        
        
 



    


   


    








