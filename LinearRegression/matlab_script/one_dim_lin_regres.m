function [w] = one_dim_lin_regres(turkish_dataset)
%least squares solution 
%first column of dataset = x1,...xn 
%second column of dataset = t1,..tn

[n, c] = size(turkish_dataset); 


x = turkish_dataset(:,1); 
t = turkish_dataset(:,2); 

w = sum(x.*t) / sum(x.^2); 

end


