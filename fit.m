function [Err, Epo, B_min_error] = fit(L,alpha,target_mse, Max_Epoch, Min_Error, Min_Error_Epoch, X, Y) 

% L=[4 4 4 3];   % // Defining the layers: Total of 4 layers, # of nodes are 2, 4, 4, 1 respectively from input to output layer
% alpha = 0.2;   % //usually alpha < 0, ranging from 0.1 to 1
% target_mse=0.05 % // one of the exit condition
% Max_Epoch=10  % // one of the exit condition
% Min_Error=Inf
% Min_Error_Epoch=-1
epoch=0;       % // 1 epoch => One forward and backward sweep of the net for each training sample 
mse =Inf;      % // initializing the Mean Squared Error with a very large value.
Err=[];
Epo=[];




[Nx,P]=size(X); % // Nx = # of sample in X, P= # of feature in X
[Ny, K]=size(Y); % // Ny = # of target output in Y, K= # of class for K classes when K>=3 otherwise, K=1 (for Binary case)

% Optional: Since input and output are kept in different files, it is better to verify the loaded sample size/dimensions.
if Nx ~= Ny 
      error ('The input/output sample sizes do not match');
end

% Optional
if L(1) ~= P
       error ('The number of input nodes must be equal to the size of the features')
end 

% Optional
if L(end) ~= K
       error ('The number of output node should be equal to K')
end 

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

B=cell(length(L)-1,1);  % forming the number of Beta/weight matrix needed in between the layers
for i=1:length(L)-1
	B{i} =rand(L(i + 1), L(i) + 1)'; % populate initial weight
end


B_min_error = B;

%Let us allocate places for error term delta, d
d=cell(length(L),1);
for i=1:length(L)
	d{i} =zeros(L(i),1);
end


while (mse > target_mse) && (epoch < Max_Epoch)   % outer loop with exit conditions
  
  CSqErr=0; 		% //Cumulative Sq Err of each Sample; we will take the average after computing Nx_th sample (=> mse)

  for j=1:Nx 		    % // for loop #1		
      Z{1} = [X(j,:) 1]';   % // Load Inputs with bias=1
      Yk   = Y(j,:)'; 	    % // Load Corresponding Desired or Target output
      % forward propagation 
      % ----------------------
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
      end  % // end of forward propagation 
         
      CSqErr= CSqErr+sum((Yk-Z{end}).^2);  % // collect sample wise Cumulative Sq Err      
     % // Compute error term delta 'd' for each of the node except the input unit
     % -----------------------------------------------------------------------
     d{end}=(Z{end}-Yk).*Z{end}.*(1-Z{end}); % // delta error term for the output layer
    
       for i=length(L)-1:-1:2 
          d{i}=Z{i}(1:end-1).*(1-Z{i}(1:end-1)).*sum(d{i+1}'*B{i}(1:end-1,:)'); % //compute the error term for all the hidden layer (and skip the input layer).
       end              

       % Now we will update the parameters/weights
       for i=1:length(L)-1 
          B{i}(1:end-1,:)=B{i}(1:end-1,:)-alpha.*(Z{i}(1:end-1)*d{i+1}'); 
          B{i}(end,:)=B{i}(end,:)-alpha.*d{i+1}';  			% // update weight connected to the bias unit(or, intercept)	
       end              
       
       
  end  % //end of for loop #1
    
    CSqErr= (CSqErr) /(Nx);        % //Average error of N sample after an epoch 
    mse=CSqErr 
    epoch  = epoch+1
    
    Err = [Err mse];
    Epo = [Epo epoch];   


    if mse < Min_Error
	Min_Error=mse;
        Min_Error_Epoch=epoch;
        B_min_error = B;
    end 

        
					    	
end % //while_end

      Min_Error
      Min_Error_Epoch     
end %end of method fit

