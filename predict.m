function [ Y_pred ] = predict( L, X, B)
%PREDICT Summary of this function goes here
%   Detailed explanation goes here
%   
% 
%   Y_pred = Predicted output
%   X = feature
%   B = Weights
% 
% %// ================================================================================================================
% % ///////////////////////////////////////////////// Predict output based on learned weight  ////////////////////////////////////////////////
% % // Here I will be using the last B computed to demo test data to classify but you should save and use best B. 
% % // Feed forward part will actually be used, assume test points: 1.(0.5,0.3) and 2.(5,4)
% % // NOTE: For point (1) the output is expected to be close to zero
% % //      For point (2) the output is expected to be close to one.
% 
% 

[Nx,P]=size(X); % // Nx = # of sample in X, P= # of feature in X

Y_pred = zeros(Nx, L(end));

%Let us allocate places for Term, T 
T=cell(length(L),1);
for i=1:length(L)
	T{i} =ones (L(i),1);
end

%Let us allocate places for activation, i.e., Z
Z=cell(length(L),1);

Z{1} = zeros (L(1) + 1,1);

for i=2:length(L)-1
	Z{i} =zeros (L(i),1); % it does not matter how do we initialize (with '0' or '1', or whatever,) this is fine!
end

Z{end} =zeros (L(end),1);  % at the final layer there is no Bias unit

  
%// ====== Same (or similar) code as we used before for feed-forward part (see above)
  for j=1:Nx 		    % for loop #1		
      Z{1} = [X(j,:) 1]';   % Load Inputs with bias=1
      %%% //(Note: desired output here) .....  Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
  
      % // forward propagation 
      % //----------------------
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
      end  % //end of forward propagation 
      Y_pred(j, :) = Z{end};
   end 

end

