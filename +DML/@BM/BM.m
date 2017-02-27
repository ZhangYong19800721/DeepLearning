classdef BM
    %BM 
    %   Detailed explanation goes here
    
    properties
        numOfVisual; %����Ԫ�ĸ���
        numOfHidden; %����Ԫ�ĸ���
        numOfUnit; %����Ԫ�ĸ���
        stateUnit; %������Ԫ��״̬
        weight; %Ȩֵ����
        bias; %ƫ��ֵ
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
        examples = generate(obj,numOfExamples) % ��RBM���Բ���numOfExamples����������
    end
    
    methods (Access = private)
        [deltaWeight,deltaBias] = update(obj, visual0) % The training step for BM
    end
end

