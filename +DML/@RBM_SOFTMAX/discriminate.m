function type = discriminate(obj,data)
%DISCRIMINATE
%   
    example_num = size(data,2);
    type = -1 * ones(1,example_num);
    softmax_part = 1:obj.num_softmax;
    visual_part = (obj.num_softmax+1):obj.num_visual;
    softmax_set = eye(obj.num_softmax);
    
    for example_index = 1:example_num
        obj.visual_unit(visual_part,:) = data(:,example_index);
        minimum_free_energy = 1e50;
        
        for softmax_index = 1:obj.num_softmax
            softmax = softmax_set(:,softmax_index);
            obj.visual_unit(softmax_part,:) = softmax;
            % 计算该显神经元对应的自由能量
            free_energy = obj.visual_unit' * obj.visual_bias + sum(log(1 + exp(obj.weight * obj.visual_unit + obj.hidden_bias)));
            free_energy = -1 * free_energy;
            if free_energy < minimum_free_energy
                minimum_free_energy = free_energy;
                type(example_index) = softmax_index - 1;
            end
        end
    end
end

