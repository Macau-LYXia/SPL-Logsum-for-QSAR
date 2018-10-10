function [ temp ] = sortt(y_train,prob_train,R)
w_chang=y_train-prob_train;
temp = [w_chang R];
[m n] =sort(abs(temp(:,1)));
temp=temp(n,:);
temp(:,1)=abs(temp(:,1));
temp=[temp n];
end

