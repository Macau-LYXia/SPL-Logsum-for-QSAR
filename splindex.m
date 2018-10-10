function [ result ] = splindex( x_train,y_train,tbeta,x_test,y_test,actual_beta )
m= size (tbeta,2);
result = zeros(m,10);
intercept=0.0;
for i =1 : m
    beta= tbeta(:,i);
    l_test=intercept+x_test*beta;
    test_y=sigmoid(l_test);
    l_train=intercept+x_train*beta;
    train_y=sigmoid(l_train);
    [Sensitivity_train,Specificity_train,accuracy_train] = printClassMetrics (train_y, y_train);
    [Sensitivity_test,Specificity_test,accuracy_test] = printClassMetrics (test_y, y_test);
    [Sensitivity_beta , Specificity_beta]=GetFPTP(actual_beta,beta);
    auc_train = roc_curve(train_y,y_train);
    auc_test = roc_curve(test_y,y_test);
    result(i,1)=auc_train;
    result(i,2)=Sensitivity_train;
    result(i,3)=Specificity_train;
    result(i,4)=accuracy_train;
    result(i,5)=auc_test;
    result(i,6)=Sensitivity_test;
    result(i,7)=Specificity_test;
    result(i,8)=accuracy_test;
    result(i,9)=Sensitivity_beta;
    result(i,10)=Specificity_beta;
end
end

