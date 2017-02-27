classdef CNN
    %CNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        num_Input; %输入层的神经元个数
        num_C1; %第1卷积层的神经元个数
        num_S2; %第2子采样层的神经元个数
        num_C3; %第3卷积层的神经元个数
        num_S4; %第4子采样层的神经元个数
        num_C5; %第5层的神经元个数
        num_F6; %第6层的神经元个数
        num_Output; %输出层的神经元个数
    end
    
    methods
        function obj = CNN(num_Input,num_C1,num_S2,num_C3,num_S4,num_C5,num_F6,num_Output)
            obj.num_Input = num_Input;
            obj.num_C1 = num_C1;
            obj.num_S2 = num_S2;
            obj.num_C3 = num_C3;
            obj.num_S4 = num_S4;
            obj.num_C5 = num_C5;
            obj.num_F6 = num_F6;
            obj.num_Output = num_Output;
        end
    end
    
    methods
        
    end
end

