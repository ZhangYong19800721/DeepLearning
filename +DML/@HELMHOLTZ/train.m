function obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN 训练Helmholtz机
%   
    obj = obj.pretrain(train_data,learn_rate_min,learn_rate_max,max_iteration); % 使用CD1快速算法进行预训练
    %save('check_point_01.mat');
    load('check_point_01.mat');
    obj.decoder_layers = obj.encoder_layers; % 将decoder_layers初始化为与encoder_layers相同
    obj = obj.wake_sleep(train_data,learn_rate_min,learn_rate_max,max_iteration); % 使用wake_sleep算法进行训练
end

