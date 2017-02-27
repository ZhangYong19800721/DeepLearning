classdef BM
    %BM 
    %   Detailed explanation goes here
    
    properties
        numOfVisual; %显神经元的个数
        numOfHidden; %隐神经元的个数
        numOfUnit; %总神经元的个数
        stateUnit; %所有神经元的状态
        weight; %权值矩阵
        bias; %偏置值
    end
    
    methods
        function obj = BM(numOfHidden,numOfVisual)
            obj.numOfVisual = numOfVisual;
            obj.numOfHidden = numOfHidden;
            obj.numOfUnit = obj.numOfVisual + obj.numOfHidden;
            obj.weight = 0.01 * (rand(obj.numOfUnit,obj.numOfUnit) - 0.5);
            obj.weight = obj.weight .* (ones(obj.numOfUnit,obj.numOfUnit) - eye(obj.numOfUnit));
            obj.weight = triu(obj.weight);
            obj.weight = obj.weight + obj.weight';
            obj.bias = 0.01 * (rand(obj.numOfUnit,1) - 0.5);
            obj.stateUnit = ones(obj.numOfUnit,1);
        end
        
        obj = train(obj,visual0,learningRate) % Use the CD1 method to update the weights and bias of the RBM.
        examples = generate(obj,numOfExamples) % 让RBM独自产生numOfExamples个数的样本
    end
    
    methods (Access = private)
        [deltaWeight,deltaBias] = update(obj, visual0) % The training step for BM
    end
end

