function data = decode(obj,code)
%DECODE 给定编码，计算其数据
%   此处显示详细说明
    num_of_layers = length(obj.decoder_layers);
    for n = num_of_layers:-1:1
        code = obj.decoder_layers(n).rbm.likelihood(code);
    end
    data = code;
end

