classdef MLP
    %MLP Multi-Layer Perceptron ����֪����
    %   Detailed explanation goes here
    
    properties
        layers; %����һ��cell array����������֪���Ķ����
    end
    
    methods
        function obj = MLP(num_layers) %���캯��
            % ���������num_layersָ���˶���֪���Ĳ���,
            obj.layers = cell(1,num_layers);
        end
    end
    
    methods
        
    end
end

