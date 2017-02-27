function obj = initialize_sgd(obj,train_data)
%INITIALIZE_SGD ��ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
%   
    [num_visual,minibatch_size,minibatch_num] = size(train_data);
    minibatch_sum = zeros(num_visual,minibatch_size); 
    
    for minibatch_index = 1:minibatch_num
        minibatch = train_data(:,:,minibatch_index);
        minibatch_sum = minibatch_sum + minibatch;
    end
    
    minibatch_sum = minibatch_sum / minibatch_num;
    obj = obj.initialize(minibatch_sum);
end

