function hidden = posterior_sample(obj,visual)
%POSTERIOR_SAMPLE �ڸ����Բ���Ԫȡֵ������£�������Ԫ���г���
%  
    num_example = size(visual,2);
    hidden = DML.sample(DML.sigmoid(obj.weight * visual + repmat(obj.hidden_bias,1,num_example)));
end

