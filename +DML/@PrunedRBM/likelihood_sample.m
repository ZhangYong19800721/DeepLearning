function visual = likelihood_sample(obj,hidden) 
%LIKELIHOOD_SAMPLE 在给定隐层神经元取值的情况下，对显神经元进行抽样
%   
    num_example = size(hidden,2);
    visual = DML.sample(DML.sigmoid(obj.weight'* hidden + repmat(obj.visual_bias,1,num_example)));
end

