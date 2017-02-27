function obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN 训练SAE
%   
    [~,minibatch_size,minibatch_num] = size(train_data);
    
    for n = 1:length(obj.layers)
        screen_message = strcat(strcat('Initialize ' , num2str(n)), ' layer RBM ......')
        obj.layers(n).rbm = obj.layers(n).rbm.initialize_sgd(train_data); % 初始化第n层的RBM
        screen_message = strcat(strcat('Initialize ' , num2str(n)), ' layer RBM ...... ok!')
        
        screen_message = strcat(strcat('Train ' , num2str(n)), ' layer RBM ......')
        obj.layers(n).rbm = obj.layers(n).rbm.pretrain_sgd(train_data,learn_rate_min,learn_rate_max,max_iteration); % 训练第n层的RBM
        screen_message = strcat(strcat('Train ' , num2str(n)), ' layer RBM ...... ok!')
   
        screen_message = 'Map data to upper layer ......'
        mapped_train_data = zeros(obj.layers(n).rbm.num_hidden,minibatch_size,minibatch_num);
        for minibatch_index = 1:minibatch_num %将训练数据映射到上一层
            mapped_train_data(:,:,minibatch_index) = obj.layers(n).rbm.posterior(train_data(:,:,minibatch_index));
        end
        train_data = mapped_train_data;
        screen_message = 'Map data to upper layer ...... ok!'
    end
end

