function obj = initialize_whv( obj,weight,hidden_bias,visual_bias )
%INITIALIZE_WHV 使用权值、隐神经元偏置、显神经元偏置直接初始化RBM
%   此处显示详细说明
    [obj.num_hidden, obj.num_visual] = size(weight);
    obj.weight = weight;
    obj.hidden_bias = hidden_bias;
    obj.visual_bias = visual_bias;
    obj.hidden_unit = zeros(obj.num_hidden,1);
    obj.visual_unit = zeros(obj.num_visual,1);
end

