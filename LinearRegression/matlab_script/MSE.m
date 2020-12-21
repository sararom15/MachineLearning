function [objective] = MSE(t,x,w_1,w_0,model)
%Compute the Mean Square Error 
%J=1/N*Sum((ti-yi)^2) with i=1:n

n = length(t); 
if model == 1 %one dimensional linear regression problem 
    %compute y: 
    y = w_1*x; 
    objective = 0; 
    for i = 1:n
        objective = objective + (t(i)-y(i))^2; 
    end
    objective = objective/n; 
end

if model == 2 %one dimensional linear regression problem with offset
    y = w_0 + w_1*x; 
    objective = 0; 
    for i = 1:n
        objective = objective + (t(i)-y(i))^2; 
    end
    objective = objective/n; 
    
end

if model == 3 %multidimensional linear regression problem 
    y = x*w_1; 
    objective = immse(t,y); 
end


end

