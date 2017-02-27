function hidden = posterior(obj,visual)
%POSTERIOR 
%      
    num_example = size(visual,2);
    hidden = DML.sigmoid(obj.weight * visual + repmat(obj.hidden_bias,1,num_example));
end

