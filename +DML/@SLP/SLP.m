classdef SLP
    %SLP 单层感知器的类
    %   单层感知器（多层感知器的组成原件）
    
    properties
        num_input; %输入神经元的个数
        num_output; %输出神经元的个数
        
        input_unit; %输入神经元的状态（维度为：num_input * 1）
        output_unit; %输出神经元的状态（维度为：num_output * 1）
        
        weight; %权值矩阵（维度为：num_output * num_input）
        bias; %输出神经元的权值（维度为：num_output * 1）
    end
    
    methods
        function obj = SLP(num_input,num_output) % 构造函数
            obj.num_input = num_input;
            obj.num_output = num_output;
            
            obj.input_unit = zeros(obj.num_input,1); %初始化输入神经元的状态为全0
            obj.output_unit = zeros(obj.num_output,1); %初始化输出神经元的状态为全0
            
            obj.weight = zeros(obj.num_output,obj.num_input); %初始化权值矩阵为全0
            obj.bias = zeros(obj.num_output,1); %初始化偏置值矩阵为全0
        end
    end
    
    methods
        output = Forward(obj, input) %前向传播信号，即根据输入计算输出
    end
end

