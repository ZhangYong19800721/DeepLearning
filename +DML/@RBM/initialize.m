function obj = initialize(obj,train_data)
    %INTIALIZE ��ʼ��Ȩֵ����Ϊ0������С���������ʼ���Բ���Ԫ��ƫ��Ϊ������ʣ���ʼ��������Ԫ��ƫ��Ϊ0.
    obj.weight = 0.01 * randn(size(obj.weight));
    x = sum(train_data,2) / size(train_data,2);
    x(x<=0) = x(x<=0) + 0.000001;
    x(x>=1) = x(x>=1) - 0.000001;
    obj.visual_bias = log(x./(1-x));
    obj.hidden_bias = zeros(size(obj.hidden_bias));
end

