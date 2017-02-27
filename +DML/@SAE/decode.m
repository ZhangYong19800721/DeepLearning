function data = decode(obj,code,layers)
%DECODE 给定编码，计算其数据
%   此处显示详细说明

    for n = layers:-1:2
        code = obj.rbm_layers(n).rbm.likelihood(code);
    end
    data = obj.rbm_layers(1).rbm.likelihood(code);
end

