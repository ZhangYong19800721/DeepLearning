function obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN 训练Helmholtz机
%   
    obj = obj.pretrain(train_data,learn_rate_min,learn_rate_max,max_iteration); % 使用CD1快速算法进行预训练
    
%     temp = obj;
%     save('check_point_4_layer784x4096x2048x1024x512.mat');
%     temp.encoder_layers = obj.encoder_layers;
%     temp.decoder_layers = obj.decoder_layers;
%     obj = temp;
%     clear temp;
    
    obj.decoder_layers = obj.encoder_layers; % 将decoder_layers初始化为与encoder_layers相同
    obj = obj.wake_sleep(train_data,learn_rate_min,learn_rate_max/10,max_iteration); % 使用wake_sleep算法进行训练
end

