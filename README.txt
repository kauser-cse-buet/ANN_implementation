- Go to project directory. 
- Make sure all the following files in the current directory: 
1.	evaluate_ann_model.m
2.	cross_validate.m
3.	fit.m
4.	predict.m
5.	get_error.m
6.	get_class_label.m
7.	get_f1_score.m

- Call function evaluate_ann_model in Matlab console as follow: 
n_hidden_layer = 1
epoch = 200
cv = 10
evaluate_ann_model( n_hidden_layer, epoch, cv)
- make sure data is present in directory "Data_and_Info".

