function [ beta ] = spl(x_train,y_train )
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
            ee=temp*0.1;
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
end

