function code = encode_vbr(obj,data)
%ENCODE_VBR 此处显示有关此函数的摘要
%   此处显示详细说明
    B=0.05; T=0.95;
    layer(1).data = data;
    for n = 1:length(obj.rbm_layers)
        layer(n+1).data = obj.rbm_layers(n).rbm.posterior(layer(n).data);
        layer(n+1).code = obj.rbm_layers(n).rbm.posterior_sample(layer(n).data);
    end
    
    code = layer(length(obj.rbm_layers)+1).code;
    layer(length(obj.rbm_layers)+1).eata = layer(length(obj.rbm_layers)+1).code; 
    for n = (length(obj.rbm_layers)+1):-1:3
        layer(n-1).eata = obj.rbm_layers(n-1).rbm.likelihood(layer(n).eata);
        index = B < layer(n-1).eata & layer(n-1).eata < T;
        code  = [code; layer(n-1).code(index)];
        layer(n-1).eata(index) = layer(n-1).code(index);
    end
end

