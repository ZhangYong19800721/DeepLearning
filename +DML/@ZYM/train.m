function obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN ѵ��Helmholtz��
%   
    obj = obj.pretrain(train_data,learn_rate_min,learn_rate_max,max_iteration); % ʹ��CD1�����㷨����Ԥѵ��
    
%     temp = obj;
%     save('check_point_4_layer784x4096x2048x1024x512.mat');
%     temp.encoder_layers = obj.encoder_layers;
%     temp.decoder_layers = obj.decoder_layers;
%     obj = temp;
%     clear temp;
    
    obj.decoder_layers = obj.encoder_layers; % ��decoder_layers��ʼ��Ϊ��encoder_layers��ͬ
    obj = obj.wake_sleep(train_data,learn_rate_min,learn_rate_max/10,max_iteration); % ʹ��wake_sleep�㷨����ѵ��
end

