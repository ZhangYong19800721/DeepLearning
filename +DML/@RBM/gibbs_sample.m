function obj = gibbs_sample( obj, times )
%SAMPLE Summary of this function goes here
%   Detailed explanation goes here
    for n = 1:times
        obj.visual_unit = DML.sample(DML.sigmoid(obj.weight'* obj.hidden_unit + obj.visual_bias));
        obj.hidden_unit = DML.sample(DML.sigmoid(obj.weight * obj.visual_unit + obj.hidden_bias));
    end
end

