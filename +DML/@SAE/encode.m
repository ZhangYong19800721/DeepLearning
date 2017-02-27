function code = encode(obj, data, layers)
%ENCODE 给定数据，计算其编码 
%   
    for n = 1:(layers-1)
        data = obj.rbm_layers(n).rbm.posterior(data);
    end
    code = obj.rbm_layers(layers).rbm.posterior_sample(data);
end

