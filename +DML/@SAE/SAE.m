classdef SAE
    %SAE Stacked Auto Encoder, 栈式自动编码器
    %   此处显示详细说明
    
    properties
        layers;
    end
    
    methods
        function obj = SAE(config)
            number_of_layers = length(config) - 1;
            for index = 1:number_of_layers
                obj.layers(index).rbm = DML.RBM(config(index+1),config(index));
            end
        end
    end
    
    methods
        obj = train(obj,directory,filename,minibatch_num,learn_rate_min,learn_rate_max,max_iteration) % 使用CD1快速算法，逐层训练DBN
        code = encode(obj,data,layers) % 给定数据，计算其编码 
        data = decode(obj,code,layers) % 给定编码，计算其数据
        
        code = encode_vbr(obj,data) % 给定数据，进行变长编码
        data = decode_vbr(obj,code) % 给定编码，恢复原始数据
    end
end

