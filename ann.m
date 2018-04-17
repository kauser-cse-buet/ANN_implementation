% T. Hoque: Here we are solving our popular Cancer{Benign, Malignant} toy problem using ANN {Back_Prop based}
% Layer_node information: L=[2 4 4 1], you can chance to see the effect.


% ///////////////////////////////////////////////// Training Section ////////////////////////////////////////////////
% ////////// initialization



% 
% %//=============================================================================================================================
% %//The NN Node and structure needs to be saved, i.e. save L
%     L
% 	
% %// Now the predicted weight B with least error should be saved in a file to be loaded and to be used for test set/new prediction
% 
%   for i=max(size(B))
%       B{i}
%   end
%   
  

  
%   
%   
%   
% 
% %// ================================================================================================================
% % ///////////////////////////////////////////////// Test Section ////////////////////////////////////////////////
% % // Here I will be using the last B computed to demo test data to classify but you should save and use best B. 
% % // Feed forward part will actually be used, assume test points: 1.(0.5,0.3) and 2.(5,4)
% % // NOTE: For point (1) the output is expected to be close to zero
% % //      For point (2) the output is expected to be close to one.
% 
% 
%   X=[0.5 0.3 0.5 0.3; 0.7 0.4 0.6 0.1]
%   
% %// ====== Same (or similar) code as we used before for feed-forward part (see above)
%   for j=1:2 		    % for loop #1		
%       Z{1} = [X(j,:) 1]';   % Load Inputs with bias=1
%       %%% //(Note: desired output here) .....  Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
%   
%       % // forward propagation 
%       % //----------------------
%       for i=1:length(L)-1
%        	     T{i+1} = B{i}' * Z{i};
%             
%              if (i+1)<length(L)
%                Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
%              else  
%                Z{i+1}=(1./(1+exp(-T{i+1}))); 
%              end 
%       end  % //end of forward propagation 
%        z
%    end 
% 
% %===============================================================================================================
% %plot epoch versus error graph
% plot (Epo,Err)  % plot based on full epoch
% 
% % plot (Epo(1:200),Err(1:200)) 
%   




FID = fopen('Data_and_Info/iris.data.txt');
C_data0 = textscan(FID,'%f %f %f %f %s', 200, 'Delimiter',',');
X = cell2mat(C_data0(:,1:4)); %ignores the last column of strings
[Nx,P]=size(X); % // Nx = # of sample in X, P= # of feature in X
target = C_data0(:,5);
class_values = target{1};
Y = ones(length(class_values), 3);
[Ny, K]=size(Y); % // Ny = # of target output in Y, K= # of class for K classes when K>=3 otherwise, K=1 (for Binary case)

for i = 1: Ny
    if strcmp(class_values{i}, 'Iris-setosa')
        Y(i, :) = [1 0 0];
    end
    if strcmp(class_values{i}, 'Iris-versicolor')
        Y(i, :) = [0 1 0];
    end
    if strcmp(class_values{i}, 'Iris-virginica')
        Y(i, :) = [0 0 1];
    end
end

% [Err, Epo, B_min_error] = fit(L,alpha,target_mse, Max_Epoch, Min_Error, Min_Error_Epoch, X, Y);
% [ Y_pred ] = predict( L, [5.1,3.5,2.4,0.2; 5.1,3.5,1.4,0.3], B_min_error);
% [ avg_mse_error_rate, avg_class_error_rate ] = get_error( Y_pred, [0 0 1; 1 0 0]);

epoch = 20; 
cv = 10;
cv_test_err_list = zeros(epoch, 1);
cv_train_err_list = zeros(epoch, 1);
L=[4 4 4 3];   % // Defining the layers: Total of 4 layers, # of nodes are 2, 4, 4, 1 respectively from input to output layer
for i = 1: epoch
    [ cv_test_err, cv_train_err ] = cross_validate( X, Y, cv, L);
    cv_test_err_list(i) = cv_test_err;
    cv_train_err_list(i) = cv_train_err;
end

plot(1:epoch, cv_test_err);
