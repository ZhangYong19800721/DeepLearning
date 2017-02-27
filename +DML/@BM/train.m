function obj = train( obj,visual0,learningRate )
%TRAIN 
%   Detailed explanation goes here
    deltaWeight = ones(size(obj.weight));
    deltaBias = ones(size(obj.bias));
    
    stopCriteia = 1e-3;
    x1 = max(max(abs(deltaWeight)));
    x2 = max(abs(deltaBias));
    x3 = max([x1,x2]);
    
    n = 0;
    
    while x3> stopCriteia && n < 5e3
        [deltaWeight deltaBias] = update(obj,visual0);
        obj.weight = obj.weight + learningRate * deltaWeight;
        obj.bias = obj.bias + learningRate * deltaBias;
        
        x1 = max(max(abs(deltaWeight)));
        x2 = max(abs(deltaBias));
        x3 = max([x1,x2])
        
        n = n + 1
    end
end

