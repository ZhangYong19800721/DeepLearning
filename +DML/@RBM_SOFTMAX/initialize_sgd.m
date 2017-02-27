function obj = initialize_sgd(obj,directory,filename,minibatch_num,below_layers)
%INITIALIZE_SGD 初始化权值矩阵，隐神经元偏置和显神经元偏置
%   
    minibatch_size = 100;
    minibatch_sum = zeros(obj.num_visual - obj.num_softmax,minibatch_size);
    num_below_layers = length(below_layers);
    
    for minibatch_index = 1:minibatch_num
        minibatch_filename = strcat(directory,strcat(filename,strcat('_',strcat(num2str(minibatch_index),'.txt'))));
        minibatch = importdata(minibatch_filename);
        if num_below_layers ~= 0 %当下面的层数不等于0时，要进行数据映射
            for layer_index = 1:num_below_layers
                minibatch = below_layers(layer_index).posterior(minibatch);
            end
        end
        minibatch_sum = minibatch_sum + minibatch;
    end
    
    minibatch_sum = minibatch_sum / minibatch_num;
    labels = repmat(eye(obj.num_softmax),1,minibatch_size/obj.num_softmax);
    minibatch_sum = [labels;minibatch_sum];
    obj = obj.initialize(minibatch_sum);
end

