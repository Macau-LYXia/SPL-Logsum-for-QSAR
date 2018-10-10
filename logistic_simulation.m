clear
clc
%%%%%%%%%% generate benchmark dataset %%%%%%%%%%%%%%%%%%%%%%%%
beta=zeros(1,1000);
beta(1)=1;
beta(2)=1;
beta(3)=-1;
beta(4)=-1;
beta(5)=1;
beta(6)=1;
beta(7)=-1;
beta(8)=1;
beta(9)=-1;
beta(10)=1;
actual_beta=beta';
%%%%%%%%the size of train and test%%%%
sample_size=300;
test_size=floor(sample_size*0.3);
train_size=sample_size-test_size;
%%%%%%%%      B0  %%%%%%%%%%%%%%%
intercept=0.0;
ii=1;
%%%%% produce the data%%%%
X = normrnd(0, 1, sample_size, size(beta,2)+1);
[n,p]=size(X);
% correlation
cor=0.2;
for i=2:5
    X(:,i) = cor*(X(:,1))+ (1-cor)*(X(:,i));
end
x=X(:,1:p-1);
%%%%control signla %%%%
l=intercept+(x*beta'+0.3*normrnd(0, 1, n, 1));
y=sigmoid(l);

%%%%%rand the sort of sample
R = randperm(n)';

%%%%% divid training and test %%%%
x_test = x(R(1:test_size),:);
y_test=y(R(1:test_size),:);
R(1:test_size)=[];
RR= R;
x_train=x(R,:);
y_train=y(R,:);
y_train(R(1:2))=inverse(y_train(R(1:2)));
xx_trian=x_train;
yy_trian=y_train;

%%%%%%%%% B0 %%%%%%%%%%%%
[row, col] = size(x_train);
temp=sum(y_train)/row;
beta_zero=log(temp/(1-temp));    %intercept
beta=zeros(col,1);

%%%%%%%%%%%%%% compute lambda on the log scale %%%%%%%%%%%%%%%%%%%%%
eta=beta_zero;   %%%%% eta= intercept + X*beta;
Pi=exp(eta)./(1+exp(eta));
W=diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%
r=(W^-1)*(y_train-Pi);            %residual= (w^-1)*(y_train-pi)
S=(x_train'*W*r)/row;

lambda_max=(4/3*(max(S)))^(1.5);
lambda_min = lambda_max*0.00005;
m =10;
for i=1:m
    Lambda(i) = lambda_max*(lambda_min/lambda_max)^(i/m);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:m
    lambda=Lambda(i);
    iter=0;
    maxiter=1000;
    beta_path(:,i)=beta(:,1);
    while iter<maxiter %true
        beta_temp=beta;
        beta_zero_temp=beta_zero;
        eta=beta_zero_temp+x_train*beta_temp;   %%%%% eta= intercept + X*beta;
        Pi=exp(eta)./(1+exp(eta));
        W=diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%
        r=(W^-1)*(y_train-Pi);            %residual= (w^-1)*(y_train-pi)
        %%%%%%%%%%%%%%%%%%%% intercept%%%%%%%%%%%%%%%%%%%%%%
        beta_zero=sum(W*r)/sum(sum(W))+beta_zero_temp;    %%%
        r=r-(beta_zero-beta_zero_temp);                %%%%%%%%%%%%%%%%%%%%%%
        for j=1:col
            v=x_train(:,j)'*W*x_train(:,j)/row;
            S=(x_train(:,j)'*W*r)/row+beta_temp*v;           %%%%%%% v is weight
            %%%%%%%%%%%%%%%%%% Soft Thresholding(Dohono,et al 1995) %%%%%%%%%%%%%%%%%
            temp=min(sqrt(lambda),lambda/abs(S(j)));
            ee=temp*0.05;
            c1=abs(S(j))-ee;
            c2=abs(c1)^2-4*(lambda-ee*abs(S(j)));
            margin = c2;
            % Soft thresholding
            if margin > 0
                beta(j)=sign(S(j))*((c1+sqrt(c2))/2)/v;
            else
                beta(j) = 0;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%--update r---%%%%%%%%%%%%%%%%%%%%%%%%%%
            r=r-x_train(:,j)*(beta(j)-beta_temp(j));
        end
        if norm(beta_temp - beta) < (1E-5)
            break;
        end
        iter=iter+1;
    end
end
[opt,Mse]=CV_logistic(x_train,y_train,Lambda,beta_path);
beta=beta_path(:,opt);
l_train=intercept+xx_trian*beta;
prob_train = 1./(1 + exp(-l_train));
[ temp ] = sortt(y_train,prob_train,R);
m=temp(:,1);
n=temp(:,3);
age =0.1
new_sample=n(find(m<=age));
s_index{ii}=temp(new_sample,2);
new_train_size=length(new_sample);
while new_train_size < train_size
    age=age+0.05;
    x_train=x_train(new_sample,:);
    y_train=y_train(new_sample);
    new_train_size
    newbeta = spl(x_train,y_train );
    tbeta(:,ii)=newbeta;
    ii=ii+1;
    l_train=intercept+xx_trian*newbeta;
    prob_train = 1./(1 + exp(-l_train));
    [ temp1 ] = sortt(yy_trian,prob_train,R);
    mm=temp1(:,1);
    nn=temp1(:,3);
    new_sample=nn(find(mm<=age));
    s_index{ii}=temp1(new_sample,2);
    new_train_size=length(new_sample);
    x_train=xx_trian;
    y_train=yy_trian;
    if   new_train_size == train_size
        new_train_size
        newbeta = spl(x_train,y_train );
        tbeta(:,ii)=newbeta;
    end
end
result = splindex( x_train,y_train,tbeta,x_test,y_test,actual_beta );