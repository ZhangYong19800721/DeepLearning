function type = discriminate(obj,data)
%DISCRIMINATE 
%   
    data = obj.rbm_layer_1.posterior(data);
    data = obj.rbm_layer_2.posterior(data);
    type = obj.rbm_softmax.discriminate(data);
end

