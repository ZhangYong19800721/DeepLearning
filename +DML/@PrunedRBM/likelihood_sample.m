function visual = likelihood_sample(obj,hidden) 
%LIKELIHOOD_SAMPLE �ڸ���������Ԫȡֵ������£�������Ԫ���г���
%   
    num_example = size(hidden,2);
    visual = DML.sample(DML.sigmoid(obj.weight'* hidden + repmat(obj.visual_bias,1,num_example)));
end

