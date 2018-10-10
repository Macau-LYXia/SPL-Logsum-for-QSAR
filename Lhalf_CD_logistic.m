function [beta,intercept] = Lhalf_CD_logistic(x,y,lambda,beta_ini)
%LOGISTIC Summary of this function goes here
%   Detailed explanation goes here

col=size(x,2);
row=size(x,1);

temp=sum(y)/row;
beta_zero=log(temp/(1-temp));    %intercept
intercept=beta_zero;

beta=beta_ini;%zeros(col,1);

iter=0;
maxiter=1;
while iter<maxiter %true
    
    beta_temp=beta;
    beta_zero_temp=beta_zero;
    
    eta=beta_zero_temp+x*beta_temp;   %%%%% eta= intercept + X*beta;
    Pi=exp(eta)./(1+exp(eta));
    W=diag(Pi.*(1-Pi));         %%%%%%%%% W is diagonal matrix%%%%%%%%%%
    r=(W^-1)*(y-Pi);          %residual= (w^-1)*(y-pi)
    
    %%%%%%%%%%%%%%%%%%%% intercept%%%%%%%%%%%%%%%%%%%%%%
    beta_zero=sum(W*r)/sum(sum(W))+beta_zero_temp;
    r=r-(beta_zero-beta_zero_temp);                %%%%%%%%%%%%%%%%%%%%%%
    
    for j=1:col
        v=x(:,j)'*W*x(:,j)/row;
        v=1;
        S=(x(:,j)'*W*r)/row+beta_temp*v;           %%%%%%% v is weight?????
        
        %%%%%%%%%%%%%%%%%% Half Thresholding(Xu,et al 2010) %%%%%%%%%%%%%%%%%
        
%         if abs(S(j)) > ((3/4)*(lambda^(2/3)))
%             phi = acos(lambda/8*((abs(S(j))/3)^(-1.5)));
%             beta(j) = real(2/3*S(j)*(1 + cos(2/3*(pi - phi))))/v; %%%%%%%pi=3.14135....
%         else
%             beta(j) = 0;
%         end
%         
        
        
        if S(j) > lambda
            beta(j) = (S(j) - lambda)/v;
        elseif S(j) < -lambda
            beta(j) = (S(j) + lambda)/v;
        elseif abs(S(j)) <= lambda
            beta(j,1) = 0;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%--update r---%%%%%%%%%%%%%%%%%%%%%%%%%%
        r=r-x(:,j)*(beta(j)-beta_temp(j));
        
    end
    
    if norm(beta_temp - beta) < (1E-5)
        break;
    end
    
    iter=iter+1;
    
end

end

