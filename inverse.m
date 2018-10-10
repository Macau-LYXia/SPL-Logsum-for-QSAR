function b=inverse(a)
m=size(a);
for i=1:m
    if a(i)==0
        b(i)=1;
    else
        b(i)=0;
    end
end