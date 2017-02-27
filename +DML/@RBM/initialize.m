function obj = initialize(obj,train_data)
    %INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
    obj.weight = 0.01 * randn(size(obj.weight));
    x = sum(train_data,2) / size(train_data,2);
    x(x<=0) = x(x<=0) + 0.000001;
    x(x>=1) = x(x>=1) - 0.000001;
    obj.visual_bias = log(x./(1-x));
    obj.hidden_bias = zeros(size(obj.hidden_bias));
end

