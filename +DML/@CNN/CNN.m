classdef CNN
    %CNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        num_Input; %��������Ԫ����
        num_C1; %��1��������Ԫ����
        num_S2; %��2�Ӳ��������Ԫ����
        num_C3; %��3��������Ԫ����
        num_S4; %��4�Ӳ��������Ԫ����
        num_C5; %��5�����Ԫ����
        num_F6; %��6�����Ԫ����
        num_Output; %��������Ԫ����
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

