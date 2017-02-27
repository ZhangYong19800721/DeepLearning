function examples = generate(obj,numOfExamples)
%GENERATE
%   Detailed explanation goes here
    examples = -1 * ones(obj.numOfVisual,numOfExamples);
    N1 = 1e3;

    % to equilibrium
    for n1 = 1:N1
        for index = 1:obj.numOfUnit
            obj.stateUnit(index) = double(DML.sample(DML.sigmoid(obj.weight(index,:) * obj.stateUnit + obj.bias(index))));
        end
    end
    
    for n2 = 1:numOfExamples
        for index = 1:obj.numOfUnit
            obj.stateUnit(index) = double(DML.sample(DML.sigmoid(obj.weight(index,:) * obj.stateUnit + obj.bias(index))));
        end
        
        examples(:,n2) = obj.stateUnit(1:obj.numOfVisual);
    end
end

