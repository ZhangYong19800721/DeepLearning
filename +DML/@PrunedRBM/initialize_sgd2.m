function obj = initialize_sgd2(obj,train_data,below_layers)
%INITIALIZE_SGD2 初始化权值矩阵，隐神经元偏置和显神经元偏置
%   
    minibatch_num = size(train_data,3);
    minibatch_size = 100;  % minibatch的大小
    minibatch_sum = zeros(obj.num_visual,minibatch_size); 
    num_below_layers = length(below_layers);
    
    for minibatch_index = 1:minibatch_num
        minibatch = train_data(:,:,minibatch_index);
        if num_below_layers ~= 0 %当下面的层数不等于0时，要进行数据映射
            for layer_index = 1:num_below_layers
                minibatch = below_layers(layer_index).posterior(minibatch);
            end
        end
        minibatch_sum = minibatch_sum + minibatch;
    end
    
    minibatch_sum = minibatch_sum / minibatch_num;
    obj = obj.initialize(minibatch_sum);
end

