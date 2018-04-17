% function name: evaluate_ann_model
% input: 
%     n_hidden_layer = number of hidden layer
%     epoch = number of iterations
%     cv = cross validation fold number
% Output:
%     report = Matrix containg output traing and test mean square error, precision, recall, f1 score.

function [ report ] = evaluate_ann_model( n_hidden_layer, epoch, cv)
%EVALUATE_ANN_MODEL Summary of this function goes here
%   Detailed explanation goes here

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
 
cv_test_err_list = zeros(epoch, 1);
cv_train_err_list = zeros(epoch, 1);

cv_test_precision_list = zeros(epoch, K);
cv_test_recall_list = zeros(epoch, K);
cv_test_f1_score_list = zeros(epoch, K);

cv_train_precision_list = zeros(epoch, K);
cv_train_recall_list = zeros(epoch, K);
cv_train_f1_score_list = zeros(epoch, K);

n_layer = n_hidden_layer + 2;
L = zeros(n_layer, 1); % initiate L, layer arrays of neural network
L(1) = P;
L(end) = K;

for i = 2: n_layer - 1
    L(i) = randi([2, 20]);
end


folds_list = zeros(epoch, 1);
for i = 1: epoch
    [ cv_test_err, cv_train_err, cv_test_precision, cv_test_recall, cv_test_f1_score, cv_train_precision, cv_train_recall, cv_train_f1_score] = cross_validate( X, Y, cv, L);
    cv_test_err_list(i) = cv_test_err;
    cv_train_err_list(i) = cv_train_err;
    
    cv_test_precision_list(i, :) = cv_test_precision;
    cv_test_recall_list(i, :) = cv_test_recall;
    cv_test_f1_score_list(i, :) = cv_test_f1_score;
    
    cv_train_precision_list(i, :) = cv_train_precision;
    cv_train_recall_list(i, :) = cv_train_recall;
    cv_train_f1_score_list(i, :) = cv_train_f1_score;
    
    folds_list(i) = i;
end

report_filename = strcat('report_nh_', int2str(n_hidden_layer), '.csv');
report = [folds_list, cv_test_err_list, cv_train_err_list, cv_test_precision_list, cv_test_recall_list, cv_test_f1_score_list, cv_train_precision_list, cv_train_recall_list, cv_train_f1_score_list];
csvwrite(report_filename,report);

layer_filename = strcat('layer_nh_', int2str(n_hidden_layer), '.csv');
csvwrite(layer_filename,L);

end