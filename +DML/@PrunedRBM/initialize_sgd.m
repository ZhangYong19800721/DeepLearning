function obj = initialize_sgd(obj,directory,filename,minibatch_num,below_layers)
%INITIALIZE_SGD ��ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
%   
    minibatch_size = 100;  % minibatch�Ĵ�С
    minibatch_sum = zeros(obj.num_visual,minibatch_size); 
    num_below_layers = length(below_layers);
    
    for minibatch_index = 1:minibatch_num
        minibatch_filename = strcat(directory,strcat(filename,strcat('_',strcat(num2str(minibatch_index),'.txt'))));
        minibatch = importdata(minibatch_filename);
        if num_below_layers ~= 0 %������Ĳ���������0ʱ��Ҫ��������ӳ��
            for layer_index = 1:num_below_layers
                minibatch = below_layers(layer_index).posterior(minibatch);
            end
        end
        minibatch_sum = minibatch_sum + minibatch;
    end
    
    minibatch_sum = minibatch_sum / minibatch_num;
    obj = obj.initialize(minibatch_sum);
end

