function obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN ѵ��Helmholtz��
%   
    obj = obj.pretrain(train_data,learn_rate_min,learn_rate_max,max_iteration); % ʹ��CD1�����㷨����Ԥѵ��
    %save('check_point_01.mat');
    load('check_point_01.mat');
    obj.decoder_layers = obj.encoder_layers; % ��decoder_layers��ʼ��Ϊ��encoder_layers��ͬ
    obj = obj.wake_sleep(train_data,learn_rate_min,learn_rate_max,max_iteration); % ʹ��wake_sleep�㷨����ѵ��
end

