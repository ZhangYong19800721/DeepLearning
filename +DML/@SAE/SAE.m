classdef SAE
    %SAE Stacked Auto Encoder, ջʽ�Զ�������
    %   �˴���ʾ��ϸ˵��
    
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
        obj = train(obj,directory,filename,minibatch_num,learn_rate_min,learn_rate_max,max_iteration) % ʹ��CD1�����㷨�����ѵ��DBN
        code = encode(obj,data,layers) % �������ݣ���������� 
        data = decode(obj,code,layers) % �������룬����������
        
        code = encode_vbr(obj,data) % �������ݣ����б䳤����
        data = decode_vbr(obj,code) % �������룬�ָ�ԭʼ����
    end
end

