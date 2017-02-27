classdef ZYM
    %ZYM ZYM
    % ZYM机的实现
    
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
        obj = pretrain(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % 使用CD1快速算法，逐层训练
        rebuild_data = rebuild(obj,data) % 重建数据 
        rebuild_data = rebuild_sample(obj,data) % 重建数据 
        code = encode(obj,data) % 编码
        code = encode_double(obj,data) % 编码,得到的是double型的数
        data = decode(obj,code) % 解码
        obj = wake_sleep(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % 使用wake-sleep算法进行训练
        delta = wake(obj,minibatch)
        delta = sleep(obj,minibatch)
        
        code = encode_sample(obj,data) % 编码
        data = decode_sample(obj,code) % 解码
        delta = wake_sample(obj,minibatch)
        delta = sleep_sample(obj,minibatch)
    end
end

