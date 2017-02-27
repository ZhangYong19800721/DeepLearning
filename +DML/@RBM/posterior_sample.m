function hidden = posterior_sample(obj,visual)
%POSTERIOR_SAMPLE 在给定显层神经元取值的情况下，对隐神经元进行抽样
%  
    num_example = size(visual,2);
    hidden = DML.sample(DML.sigmoid(obj.weight * visual + repmat(obj.hidden_bias,1,num_example)));
end

