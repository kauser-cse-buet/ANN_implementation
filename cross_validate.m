% function name: cross_validate
% Input:
%     X = feature file, 
%     Y = target, 
%     cv = cross validation fold number, 
%     L = Layer array
% Output: 
%     cv_test_err, 
%     cv_train_err, 
%     cv_test_precision, 
%     cv_test_recall, 
%     cv_test_f1_score, 
%     cv_train_precision,
%     cv_train_recall, 
%     cv_train_f1_score
    

function [ cv_test_err, cv_train_err, cv_test_precision, cv_test_recall, cv_test_f1_score, cv_train_precision, cv_train_recall, cv_train_f1_score, B_best] = cross_validate( X, Y, cv, L, B)
%CROSS_VALIDATE Summary of this function goes here
%   Detailed explanation goes here
alpha = 0.2;   % //usually alpha < 0, ranging from 0.1 to 1

[ny, n_class] = size(Y);
y_indices = (1: ny);

CVO = cvpartition(y_indices,'k',cv);
test_err_list = zeros(CVO.NumTestSets,1);

test_precision_list = zeros(CVO.NumTestSets,n_class);
test_recall_list = zeros(CVO.NumTestSets,n_class);
test_f1_score_list = zeros(CVO.NumTestSets,n_class);

precision_train_list = zeros(CVO.NumTestSets,n_class);
recall_train_list = zeros(CVO.NumTestSets,n_class);
f1_score_train_list = zeros(CVO.NumTestSets,n_class);

train_err_list = zeros(CVO.NumTestSets,1);

B_train_list = cell(CVO.NumTestSets,1);

for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    X_train = X(trIdx, :);
    X_test = X(teIdx, :);
    Y_train = Y(trIdx, :);
    Y_test = Y(teIdx, :);
    
    
    [B_train] = fit(L,alpha, X_train, Y_train, B);
    
    B_train_list{i} = B_train;
    
    [ Y_pred_train ] = predict( L, X_train, B_train);
    [ Y_pred ] = predict( L, X_test, B_train);
    
    [ avg_mse_error_rate_train, precision_train, recall_train, f1_score_train ] = get_error( Y_pred_train, Y_train);
    [ avg_mse_error_rate, precision, recall, f1_score ] = get_error( Y_pred, Y_test);
    
    test_err_list(i) = avg_mse_error_rate;
    train_err_list(i) = avg_mse_error_rate_train;
    
    test_precision_list(i, :) = precision;
    test_recall_list(i, :) = recall;
    test_f1_score_list(i, :) = f1_score;
    
    precision_train_list(i, :) = precision_train;
    recall_train_list(i, :) = recall_train;
    f1_score_train_list(i, :) = f1_score_train;
end
cv_test_err = sum(test_err_list)/CVO.NumTestSets;
cv_train_err = sum(train_err_list)/CVO.NumTestSets;

cv_test_precision = sum(test_precision_list)/CVO.NumTestSets;
cv_test_recall = sum(test_recall_list)/CVO.NumTestSets;
cv_test_f1_score = sum(test_f1_score_list)/CVO.NumTestSets;

cv_train_precision = sum(precision_train_list)/ CVO.NumTestSets;
cv_train_recall = sum(recall_train_list)/ CVO.NumTestSets;
cv_train_f1_score = sum(f1_score_train_list)/ CVO.NumTestSets;

% find B  best and return the value. based on test err list.
B_best = B_train_list{test_err_list == min(test_err_list)};
end

