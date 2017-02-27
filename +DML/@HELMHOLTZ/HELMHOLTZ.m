classdef HELMHOLTZ
    %HELMHOLTZ Helmholtz��
    %   Helmholtz����ʵ��
    
    properties
        encoder_layers;
        decoder_layers;
    end
    
    methods
        function obj = HELMHOLTZ(config)
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
        rebuild_data = rebuild(obj,data); % �ؽ����� 
        code = encode(obj,data); % ����
        data = decode(obj,code); % ����
        obj = wake_sleep(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % ʹ��wake-sleep�㷨����ѵ��
        delta = wake(obj,minibatch)
        delta = sleep(obj,minibatch)
    end
end

