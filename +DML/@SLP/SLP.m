classdef SLP
    %SLP �����֪������
    %   �����֪��������֪�������ԭ����
    
    properties
        num_input; %������Ԫ�ĸ���
        num_output; %�����Ԫ�ĸ���
        
        input_unit; %������Ԫ��״̬��ά��Ϊ��num_input * 1��
        output_unit; %�����Ԫ��״̬��ά��Ϊ��num_output * 1��
        
        weight; %Ȩֵ����ά��Ϊ��num_output * num_input��
        bias; %�����Ԫ��Ȩֵ��ά��Ϊ��num_output * 1��
    end
    
    methods
        function obj = SLP(num_input,num_output) % ���캯��
            obj.num_input = num_input;
            obj.num_output = num_output;
            
            obj.input_unit = zeros(obj.num_input,1); %��ʼ��������Ԫ��״̬Ϊȫ0
            obj.output_unit = zeros(obj.num_output,1); %��ʼ�������Ԫ��״̬Ϊȫ0
            
            obj.weight = zeros(obj.num_output,obj.num_input); %��ʼ��Ȩֵ����Ϊȫ0
            obj.bias = zeros(obj.num_output,1); %��ʼ��ƫ��ֵ����Ϊȫ0
        end
    end
    
    methods
        output = Forward(obj, input) %ǰ�򴫲��źţ�����������������
    end
end

