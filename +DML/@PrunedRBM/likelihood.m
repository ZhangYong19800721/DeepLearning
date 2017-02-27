function visual = likelihood(obj,hidden)
%LIKELIHOOD
% 
    num_example = size(hidden,2);
    visual = DML.sigmoid(obj.weight'* hidden + repmat(obj.visual_bias,1,num_example));
end

