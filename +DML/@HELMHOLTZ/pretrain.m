function obj = pretrain(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN 训练Helmholtz机
%   
    [~,minibatch_size,minibatch_num] = size(train_data);
    num_of_layers = length(obj.encoder_layers);
    
    for n = 1:num_of_layers
        screen_message = strcat(strcat('Initialize ' , num2str(n)), ' layer RBM ......')
        obj.encoder_layers(n).rbm = obj.encoder_layers(n).rbm.initialize_sgd(train_data); % 初始化第n层的RBM
        screen_message = strcat(strcat('Initialize ' , num2str(n)), ' layer RBM ...... ok!')
        
        screen_message = strcat(strcat('Train ' , num2str(n)), ' layer RBM ......')
        obj.encoder_layers(n).rbm = obj.encoder_layers(n).rbm.pretrain_sgd(train_data,learn_rate_min,learn_rate_max,max_iteration); % 训练第n层的RBM
        screen_message = strcat(strcat('Train ' , num2str(n)), ' layer RBM ...... ok!')
   
        screen_message = 'Map data to upper layer ......'
        mapped_train_data = zeros(obj.encoder_layers(n).rbm.num_hidden,minibatch_size,minibatch_num);
        for minibatch_index = 1:minibatch_num %将训练数据映射到上一层
            mapped_train_data(:,:,minibatch_index) = obj.encoder_layers(n).rbm.posterior(train_data(:,:,minibatch_index));
        end
        train_data = mapped_train_data;
        screen_message = 'Map data to upper layer ...... ok!'
    end
end

