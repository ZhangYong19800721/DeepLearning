function data = decode(obj,code)
%DECODE �������룬����������
%   �˴���ʾ��ϸ˵��
    num_of_layers = length(obj.decoder_layers);
    for n = num_of_layers:-1:1
        code = obj.decoder_layers(n).rbm.likelihood(code);
    end
    data = code;
end

