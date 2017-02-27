function delta = sleep_sample(obj,minibatch)
%SLEEP sleep阶段
%   此处显示详细说明
    num_of_layers = length(obj.decoder_layers);
    decoder(num_of_layers).hidden_s = obj.encode_sample(minibatch);
    
    for n = num_of_layers:-1:1
        decoder(n).visual_p = obj.decoder_layers(n).rbm.likelihood(decoder(n).hidden_s);
        if n > 1
            decoder(n).visual_s = DML.sample(decoder(n).visual_p);
        else
            decoder(n).visual_s = decoder(n).visual_p;
        end
        if n > 1
            decoder(n-1).hidden_s = decoder(n).visual_s;
        end
    end
    
    encoder(1).visual_s = decoder(1).visual_s;
    for n = 1:num_of_layers
        encoder(n).hidden_p = obj.encoder_layers(n).rbm.posterior(encoder(n).visual_s);
        encoder(n).hidden_s = decoder(n).hidden_s;
        if n < num_of_layers
            encoder(n+1).visual_s = encoder(n).hidden_s;
        end
    end
    
    for n = 1:num_of_layers
        x = encoder(n).hidden_s;
        y = encoder(n).hidden_p;
        z = encoder(n).visual_s;   
        delta(n).hidden_bias = sum(x - y,2) / size(minibatch,2);
        delta(n).weight =  (z * (x - y)')' / size(minibatch,2);
    end
end

