function [deltaWeight,deltaBias] = update(obj, visual0)
%UPDATE 
%   Detailed explanation goes here
    numberOfExample = size(visual0,2);
    deltaWeight = zeros(size(obj.weight));
    weightPositive = zeros(size(obj.weight));
    weightNegative = zeros(size(obj.weight));
    deltaBias = zeros(size(obj.bias));
    biasPositive = zeros(size(obj.bias));
    biasNegative = zeros(size(obj.bias));
    N1 = 1e3;
    N2 = 1e3;

    obj.weight = obj.weight .* (ones(obj.numOfUnit,obj.numOfUnit) - eye(obj.numOfUnit));
    for n = 1:numberOfExample
        obj.stateUnit(1:obj.numOfVisual,1) = visual0(:,n); %将显示神经元绑定到训练集
        
        % to equilibrium
        for n1 = 1:N1
            for index = (obj.numOfVisual+1):obj.numOfUnit
                obj.stateUnit(index) = double(DML.sample(DML.sigmoid(obj.weight(index,:) * obj.stateUnit + obj.bias(index))));
            end
        end
        
        % 
        for n2 = 1:N2
            for index = (obj.numOfVisual+1):obj.numOfUnit
                obj.stateUnit(index) = double(DML.sample(DML.sigmoid(obj.weight(index,:) * obj.stateUnit + obj.bias(index))));
            end
            
            weightPositive = weightPositive + obj.stateUnit * obj.stateUnit';
            biasPositive = biasPositive + obj.stateUnit;
        end
    end
    
    weightPositive =  weightPositive .* (ones(obj.numOfUnit,obj.numOfUnit) - eye(obj.numOfUnit)) / (N2 * numberOfExample);
    biasPositive = biasPositive / (N2 * numberOfExample);
    
    % to equilibrium
    for n1 = 1:N1
        for index = 1:obj.numOfUnit
            obj.stateUnit(index) = double(DML.sample(DML.sigmoid(obj.weight(index,:) * obj.stateUnit + obj.bias(index))));
        end
    end
    
    for n2 = 1:N2
        for index = 1:obj.numOfUnit
            obj.stateUnit(index) = double(DML.sample(DML.sigmoid(obj.weight(index,:) * obj.stateUnit + obj.bias(index))));
        end
        
        weightNegative = weightNegative + obj.stateUnit * obj.stateUnit';
        biasNegative = biasNegative + obj.stateUnit;
    end
    
    weightNegative =  weightNegative .* (ones(obj.numOfUnit,obj.numOfUnit) - eye(obj.numOfUnit)) / N2;
    biasNegative = biasNegative / N2;
    
    deltaWeight = weightPositive - weightNegative;
    deltaWeight = triu(deltaWeight);
    deltaWeight = deltaWeight + deltaWeight';
    deltaBias = biasPositive - biasNegative;
end

