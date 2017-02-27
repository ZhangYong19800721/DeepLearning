function data = recall( obj,type )
%RECALL 
%   
    example_num = size(type,2);
    data = zeros(obj.num_visual - obj.num_softmax,example_num);
    softmax_part = 1:obj.num_softmax;
    visual_part = (obj.num_softmax+1):obj.num_visual;
    labels = eye(obj.num_softmax);
    temp_data = obj.visual_unit;
    N = 1e4;
    
    for example_index = 1:example_num
        obj.visual_unit(softmax_part,:) = labels(:,(type(example_index) + 1));
        obj.visual_unit(visual_part,:)  = zeros(length(visual_part),1);
        minimum_free_energy = 1e30;
        
        for n = 1:N
            obj.hidden_unit = DML.sample(DML.sigmoid(obj.weight  * obj.visual_unit + obj.hidden_bias));
            obj.visual_unit = DML.sample(DML.sigmoid(obj.weight' * obj.hidden_unit + obj.visual_bias));
            obj.visual_unit(softmax_part,:) = labels(:,(type(example_index) + 1));
            % 计算该显神经元对应的自由能量
            free_energy = obj.visual_unit' * obj.visual_bias + sum(log(1 + exp(obj.weight * obj.visual_unit + obj.hidden_bias)));
            free_energy = -1 * free_energy;
            if free_energy < minimum_free_energy
                minimum_free_energy = free_energy;
                temp_data = obj.visual_unit;
            end
        end
        
        data(:,example_index) = temp_data(visual_part,:);
    end
end

