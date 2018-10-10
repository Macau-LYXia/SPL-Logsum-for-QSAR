function [Sensitivity ,Specificity ]=GetFPTP(theta,theta_hat)
thea = abs(theta) > 0;    % transform coefficients to binary values
thea_hat = abs(theta_hat) > 0; % convert estimated coefficients to binary values
A = sum(~thea.*~thea_hat);  % A: TN
B = sum(~thea.*thea_hat);   % B: FP
C = sum(thea.*~thea_hat);   % C: FN
D = sum(thea.*thea_hat);    % D: TP
FPR = B/(B+A);         % FPR=FP/(FP+TN)
TPR = D/(D+C);          % TPR=TP/(TP+FN)
Sensitivity =TPR;
Specificity= 1-FPR;