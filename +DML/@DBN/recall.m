function data = recall(obj,type)
%RECALL Summary of this function goes here
%   Detailed explanation goes here
    data = obj.rbm_softmax.recall(type);
    data = obj.rbm_layer_2.likelihood(data);
    data = obj.rbm_layer_1.likelihood(data);
end

