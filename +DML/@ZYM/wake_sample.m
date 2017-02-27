function delta = wake_sample(obj,minibatch)
%WAKE wake阶段
%   此处显示详细说明
    num_of_layers = length(obj.encoder_layers);
    encoder(1).visual_p = minibatch;
    encoder(1).visual_s = encoder(1).visual_p;
    for n = 1:num_of_layers
        encoder(n).hidden_p = obj.encoder_layers(n).rbm.posterior(encoder(n).visual_s);
        encoder(n).hidden_s = DML.sample(encoder(n).hidden_p);
        if n < num_of_layers
            encoder(n + 1).visual_s = encoder(n).hidden_s;
        end
    end
    
    decoder(num_of_layers).hidden_p = repmat(DML.sigmoid(obj.decoder_layers(num_of_layers).rbm.hidden_bias),1,size(minibatch,2));
    for n = num_of_layers:-1:1
        decoder(n).hidden_s = encoder(n).hidden_s;
        decoder(n).visual_p = obj.decoder_layers(n).rbm.likelihood(decoder(n).hidden_s);
        if n < num_of_layers
            decoder(n + 1).visual_s = decoder(n).hidden_s;
        end
    end
    decoder(1).visual_s = minibatch;
    
    x = decoder(num_of_layers).hidden_s;
    y = decoder(num_of_layers).hidden_p;
    delta.top_bias = sum(x - y,2) / size(minibatch,2);
    
    for n = num_of_layers:-1:1
        x = decoder(n).hidden_s;
        a = decoder(n).visual_s;
        b = decoder(n).visual_p;
        delta(n).visual_bias = sum(a - b,2) / size(minibatch,2);
        delta(n).weight = x * (a - b)' / size(minibatch,2);
    end
end

