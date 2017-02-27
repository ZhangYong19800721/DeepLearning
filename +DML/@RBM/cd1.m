function [delta_weight,delta_hidden_bias,delta_visual_bias,recon_error] = cd1(obj, train_data)
% This is unsupervised fast training process of a RBM, adopting the method
% of Contrastive Divergence 1 (CD1).
% train_data, the input training set, each collum represets an training example, dimension (VxN).
% delta_weight, the delta weight matrix.
% delta_hidden_bias, the delta hidden bias.
% delta_visual_bias, the delta visual bias.
    example_num = size(train_data,2);
    hidden_bias = repmat(obj.hidden_bias,1,example_num);
    visual_bias = repmat(obj.visual_bias,1,example_num);
    
    hidden_field_0 = DML.sigmoid(obj.weight * train_data + hidden_bias);
    hidden_0 = DML.sample(hidden_field_0);
    visual_field_1 = DML.sigmoid(obj.weight'* hidden_0 + visual_bias);
    visual_1 = DML.sample(visual_field_1);
    hidden_field_1 = DML.sigmoid(obj.weight * visual_1 + hidden_bias);
    
    recon_error =  sum(sum(abs(visual_field_1 - train_data))) / example_num; %计算在整个mini-batch上的平均reconstruction error
    
    delta_weight = (hidden_field_0 * train_data' - hidden_field_1 * visual_1') / example_num;
    delta_hidden_bias = (hidden_0 - hidden_field_1) * ones(example_num,1) / example_num;
    delta_visual_bias = (train_data - visual_field_1) * ones(example_num,1) / example_num;
end

