function [example,obj] = generate( obj )
%GENERATE 
%   
    obj.visual_unit = DML.sample(DML.sigmoid(obj.weight'* obj.hidden_unit + obj.visual_bias));
    obj.hidden_unit = DML.sample(DML.sigmoid(obj.weight * obj.visual_unit + obj.hidden_bias));
    example = obj.visual_unit;
end

