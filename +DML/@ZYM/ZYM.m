classdef ZYM
    %ZYM ZYM
    % ZYM����ʵ��
    
    properties
        encoder_layers;
        decoder_layers;
    end
    
    methods
        function obj = ZYM(config)
            number_of_layers = length(config) - 1;
            for index = 1:number_of_layers
                obj.encoder_layers(index).rbm = DML.RBM(config(index+1),config(index));
            end
            obj.decoder_layers = obj.encoder_layers;
        end
    end
    
    methods
        obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % 
        obj = pretrain(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % ʹ��CD1�����㷨�����ѵ��
        rebuild_data = rebuild(obj,data) % �ؽ����� 
        rebuild_data = rebuild_sample(obj,data) % �ؽ����� 
        code = encode(obj,data) % ����
        code = encode_double(obj,data) % ����,�õ�����double�͵���
        data = decode(obj,code) % ����
        obj = wake_sleep(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % ʹ��wake-sleep�㷨����ѵ��
        delta = wake(obj,minibatch)
        delta = sleep(obj,minibatch)
        
        code = encode_sample(obj,data) % ����
        data = decode_sample(obj,code) % ����
        delta = wake_sample(obj,minibatch)
        delta = sleep_sample(obj,minibatch)
    end
end

