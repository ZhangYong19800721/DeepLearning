function obj = pretrain(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN ѵ��Helmholtz��
%   
    [~,minibatch_size,minibatch_num] = size(train_data);
    num_of_layers = length(obj.encoder_layers);
    
    for n = 1:num_of_layers
        screen_message = strcat(strcat('Initialize ' , num2str(n)), ' layer RBM ......')
        obj.encoder_layers(n).rbm = obj.encoder_layers(n).rbm.initialize_sgd(train_data); % ��ʼ����n���RBM
        screen_message = strcat(strcat('Initialize ' , num2str(n)), ' layer RBM ...... ok!')
        
        screen_message = strcat(strcat('Train ' , num2str(n)), ' layer RBM ......')
        obj.encoder_layers(n).rbm = obj.encoder_layers(n).rbm.pretrain_sgd(train_data,learn_rate_min,learn_rate_max,max_iteration); % ѵ����n���RBM
        screen_message = strcat(strcat('Train ' , num2str(n)), ' layer RBM ...... ok!')
   
        screen_message = 'Map data to upper layer ......'
        mapped_train_data = zeros(obj.encoder_layers(n).rbm.num_hidden,minibatch_size,minibatch_num);
        for minibatch_index = 1:minibatch_num %��ѵ������ӳ�䵽��һ��
            mapped_train_data(:,:,minibatch_index) = obj.encoder_layers(n).rbm.posterior(train_data(:,:,minibatch_index));
        end
        train_data = mapped_train_data;
        screen_message = 'Map data to upper layer ...... ok!'
    end
end

