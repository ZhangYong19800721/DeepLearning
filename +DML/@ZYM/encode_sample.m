function code = encode_sample(obj,data)
%ENCODE �������ݣ���������� 
%   
    num_of_layers = length(obj.encoder_layers);
    for n = 1:num_of_layers
        data = obj.encoder_layers(n).rbm.posterior_sample(data);
    end
    code = data;
end

