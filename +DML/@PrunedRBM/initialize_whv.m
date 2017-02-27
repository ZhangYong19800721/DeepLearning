function obj = initialize_whv( obj,weight,hidden_bias,visual_bias)
%INITIALIZE_WHV ʹ��Ȩֵ������Ԫƫ�á�����Ԫƫ��ֱ�ӳ�ʼ��RBM
%   �˴���ʾ��ϸ˵��
    [obj.num_hidden, obj.num_visual] = size(weight);
    obj.weight = weight;
    obj.mask_weight = ones(obj.num_hidden, obj.num_visual);
    obj.hidden_bias = hidden_bias;
    obj.visual_bias = visual_bias;
    obj.hidden_unit = zeros(obj.num_hidden,1);
    obj.visual_unit = zeros(obj.num_visual,1);
    obj.mask_hidden = ones(obj.num_hidden,1);
    obj.mask_visual = ones(obj.num_visual,1);
end

