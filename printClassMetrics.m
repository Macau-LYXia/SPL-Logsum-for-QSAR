function [Sensitivity,Specificity,accuracy] = printClassMetrics (pred_val , yval)

  accuracy = mean(double(pred_val == yval));
  tp = sum((pred_val == 1) & (yval == 1));
  fp = sum((pred_val == 1) & (yval == 0));
  fn= sum((pred_val == 0) & (yval == 1));
  tn = sum((pred_val == 0) & (yval == 0));
  precision = 0; 
  if ( (tp + fp) > 0)
    precision = tp / (tp + fp);
  end
  recall = 0; 
  if ( (tp + fn) > 0 )
    recall = tp / (tp + fn);
  end
  F1 = 0; 
  if ( (precision + recall) > 0) 
    F1 = 2 * precision * recall / (precision + recall);
  end
  Sensitivity = tp/(tp+fn);
  Specificity = 1-fp/(fp+tn);
  
end
