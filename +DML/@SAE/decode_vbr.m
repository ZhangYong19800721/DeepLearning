function data = decode_vbr(obj,code)
%DECODE_VBR 此处显示有关此函数的摘要
%   此处显示详细说明
    B=0.05; T=0.95; 
    num_of_layers = length(obj.rbm_layers);

    stat_pos = 1;
    stop_pos = obj.rbm_layers(num_of_layers).rbm.num_hidden;
    code_part = code(stat_pos:stop_pos);
    
    for n = num_of_layers:-1:2
        code_part = obj.rbm_layers(n).rbm.likelihood(code_part);
        index = B < code_part & code_part < T;
        stat_pos = stop_pos + 1;
        stop_pos = stop_pos + sum(index);
        code_part(index) = code(stat_pos:stop_pos);
    end
    
    data = obj.rbm_layers(1).rbm.likelihood(code_part);
end

