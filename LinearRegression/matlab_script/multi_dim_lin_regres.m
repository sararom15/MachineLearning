function [W_multi] = multi_dim_lin_regres(mpg, Xmatrix)

%firstly we have to analyze the observation matrix 
[dim.rows, dim.columns] = size(Xmatrix); 
if dim.rows == dim.columns %Xmatrix is square 
    disp('X is a square matrix'); 
    D = det(Xmatrix);
    if D~=0 %non singular matrix, then 1 and only 1 solution 
        disp('X is invertible'); 
        
        W_multi = inv_svd(Xmatrix)*mpg; 
        
        else %singular matrix 
            disp('X is a singular matrix');
        
            W_multi = pinv(Xmatrix)*mpg;%%pinv(Xmatrix) defines the pseudoinverse
    
    end
    
%normal equations for the least squares problem:
else if dim.rows > dim.columns %X is tall matrix 
        disp('X is tall matrix'); 
        
        %closed-form solution is obtained as follow: 
        W_multi = pinv(Xmatrix)*mpg; %left pseudoinverse 
    
    else  %dim.rows < dim.columns %X is flat matrix 
        disp('X is flat matrix'); 
        
        %closed-form solution is obtained as follow: 
        
        W_multi = (Xmatrix'*inv_svd(Xmatrix*Xmatrix'))*mpg;%%[Xmatrix'*inv_svd(Xmatrix*Xmatrix')] = right pseudoinverse
    end
end

end
 
 function res = inv_svd(A) 
        [U,S,V] = svd(A);
        res = transpose(V)*pinv(S)*U;
 end

        
