function [ Y_class ] = get_class_label( Y )
%GET_CLASS_LABEL Summary of this function goes here
%   Detailed explanation goes here

[ny, cn] = size(Y)
Y_class = zeros(ny, cn)
for i = 1:ny
    ind = find ( Y(i,:) == max(Y(i, :)))
    Y_class(i, ind) = 1 
end

end

