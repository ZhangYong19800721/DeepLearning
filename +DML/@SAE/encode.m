function code = encode(obj, data, layers)
%ENCODE �������ݣ���������� 
%   
    for n = 1:(layers-1)
        data = obj.rbm_layers(n).rbm.posterior(data);
    end
    code = obj.rbm_layers(layers).rbm.posterior_sample(data);
end

