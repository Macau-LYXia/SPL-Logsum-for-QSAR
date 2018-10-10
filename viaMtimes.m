function Z = viaMtimes(X,y)
%   This is an internal implementation of mpower when y is a real integer
%   scalar.
  
%   Copyright 2014 The MathWorks, Inc.

if y == 2
    Z = X*X;
elseif y == 3
    Z = X*X*X;
elseif y == 0
    Z = eye(size(X),'like',X([])); %always real
else
    % X and y can be sparse
    % Z = X^y for integer y. Use repeated squaring.
    % For example: A^13 = A^1 * A^4 * A^8
    p = abs(y);
    D = X;
    first = true;
    while p > 0
        if mod(p,2) == 1 %if odd
            if first
                Z = D;  % hit first time. D*I
                first = false;
            else
                Z = D*Z;
            end
        end
        p = fix(p/2);
        if p ~= 0
            D = D*D;
        end
    end
    if y < 0
        Z = pinv(Z);
    end
end