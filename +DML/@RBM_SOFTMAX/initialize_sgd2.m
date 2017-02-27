function obj = initialize_sgd2(obj,train_data,below_layers)
%INITIALIZE_SGD2 ��ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
%   
    minibatch_num = size(train_data,3);
    minibatch_size = 100;
    minibatch_sum = zeros(obj.num_visual,minibatch_size);
    softmax_part = 1:obj.num_softmax;
    visual_part = (obj.num_softmax+1):obj.num_visual;
    num_below_layers = length(below_layers);
    
    for minibatch_index = 1:minibatch_num
        minibatch = train_data(:,:,minibatch_index);
        if num_below_layers ~= 0 %������Ĳ���������0ʱ��Ҫ��������ӳ��
            for layer_index = 1:num_below_layers
                minibatch = below_layers(layer_index).posterior(minibatch);
            end
        end
        minibatch_sum(visual_part,:) = minibatch_sum(visual_part,:) + minibatch;
    end
    
    minibatch_sum = minibatch_sum / minibatch_num;
    labels = repmat(eye(obj.num_softmax),1,minibatch_size/obj.num_softmax);
    minibatch_sum(softmax_part,:) = labels;
    obj = obj.initialize(minibatch_sum);
end

