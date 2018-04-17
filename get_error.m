function [ avg_mse_error_rate, precision, recall, f1_score] = get_error( Y_pred, Y_actual )
%GET_ Summary of this function goes here
%   Detailed explanation goes here

Y_pred_class = get_class_label( Y_pred )
avg_mse_error_rate = mean(mean((Y_actual - Y_pred).^2, 2))
[ precision, recall, f1_score] = get_f1_score( Y_pred_class, Y_actual );

end

