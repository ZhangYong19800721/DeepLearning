function [delta_weight,delta_hidden_bias,delta_visual_bias,recon_error] = cd1(obj, train_data)
% 使用Contrastive Divergence 1 (CD1)方法对RBM进行快速训练，该RBM包含Softmax神经元。
% train_data, 训练数据，每一列是一个训练样本，每一个样本的前几个数对应softmax神经元。
% delta_weight, 权值矩阵的移动差值
% delta_hidden_bias, 隐神经元偏置的移动差值
% delta_visual_bias, 显神经元偏置的移动差值
    example_num = size(train_data,2);
    hidden_bias = repmat(obj.hidden_bias,1,example_num);
    visual_bias = repmat(obj.visual_bias,1,example_num);
    
    hidden_field_0 = DML.sigmoid(obj.weight * train_data + hidden_bias);
    hidden_0 = DML.sample(hidden_field_0);
    
    visual_field_1 = obj.weight'* hidden_0 + visual_bias;
    softmax_part = 1:obj.num_softmax;
    visual_part = (obj.num_softmax+1):obj.num_visual;
    visual_field_1(softmax_part,:) = exp(visual_field_1(softmax_part,:));
    visual_field_1(visual_part,:) = DML.sigmoid(visual_field_1(visual_part,:));
    
    softmax_field_1 = visual_field_1(softmax_part,:);
    softmax_field_1 = softmax_field_1 ./ repmat(sum(softmax_field_1),obj.num_softmax,1);
    visual_field_1(softmax_part,:) = softmax_field_1;
    
    visual_1(softmax_part,:) = DML.sample_softmax(softmax_field_1);
    visual_1(visual_part,:) = DML.sample(visual_field_1(visual_part,:));
    hidden_field_1 = DML.sigmoid(obj.weight * visual_1 + hidden_bias);
    
    recon_error =  sum(sum(abs(visual_field_1 - train_data))) / example_num; %计算在整个train_data上的平均reconstruction error
    
    delta_weight = (hidden_field_0 * train_data' - hidden_field_1 * visual_1') / example_num;
    delta_hidden_bias = (hidden_0 - hidden_field_1) * ones(example_num,1) / example_num;
    delta_visual_bias = (train_data - visual_field_1) * ones(example_num,1) / example_num;
end

