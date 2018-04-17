function [ precision, recall, f1_score] = get_f1_score( Y_pred, Y_act )
%GET_F1_SCORE Summary of this function goes here
%   Detailed explanation goes here
[ny, nclass] = size(Y_pred);

confusion_matrix = zeros(nclass, nclass);

for i = 1: ny
    row = find(Y_act(i, :), 1);
    col = find(Y_pred(i, :), 1);
    confusion_matrix(row, col) = confusion_matrix(row, col) + 1;
end

precision = zeros(nclass, 1);
recall = zeros(nclass, 1);
f1_score = zeros(nclass, 1);

for i = 1: nclass
    if sum(confusion_matrix(:, i)) == 0
        precision(i) = 0;
    else
        precision(i) = confusion_matrix(i, i) / sum(confusion_matrix(:, i));
    end
    
    if sum(confusion_matrix(i, :)) == 0
        recall(i) = 0;
    else
        recall(i) = confusion_matrix(i, i) / sum(confusion_matrix(i, :));
    end
    
    if (precision(i) + recall(i)) == 0
        f1_score(i) = 0;
    else
        f1_score(i) = 2 * precision(i) * recall(i) /(precision(i) + recall(i));
    end
end

end

