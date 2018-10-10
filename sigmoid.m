function y = sigmoid(z)

prob = 1./(1 + exp(-z));
prob(find(prob>=0.5)) =1;
prob(find(prob<0.5))=0;
y=prob;
end
