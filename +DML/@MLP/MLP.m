classdef MLP
    %MLP Multi-Layer Perceptron 多层感知器类
    %   Detailed explanation goes here
    
    properties
        layers; %这是一个cell array，包含多层感知器的多个层
    end
    
    methods
        function obj = MLP(num_layers) %构造函数
            % 输入参数：num_layers指定了多层感知器的层数,
            obj.layers = cell(1,num_layers);
        end
    end
    
    methods
        
    end
end

