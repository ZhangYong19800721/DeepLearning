function data = decode(obj,code,layers)
%DECODE �������룬����������
%   �˴���ʾ��ϸ˵��

    for n = layers:-1:2
        code = obj.rbm_layers(n).rbm.likelihood(code);
    end
    data = obj.rbm_layers(1).rbm.likelihood(code);
end

