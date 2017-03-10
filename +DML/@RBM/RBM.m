classdef RBM
    %RBM 约束玻尔兹曼机
    %   
    
    properties
        num_hidden; % The number of hidden units
        num_visual; % The number of visual units
        weight; % The weight matrix, dimension = (number of hidden units * number of visual units)
        hidden_bias; % The bias for hidden units
        visual_bias; % The bias for visual units
    end
    
    methods
        function obj = RBM(num_hidden,num_visual) % The constructor functions
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        obj = initialize_whv(obj,weight,hidden_bias,visual_bias) % 使用权值、隐神经元偏置、显神经元偏置直接初始化RBM
        obj = initialize_sgd(obj,train_data) % 初始化权值矩阵，隐神经元偏置和显神经元偏置

        obj = pretrain_sgd(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % 随机梯度下降，CD1快速算法
        
        [example, obj] = generate(obj) % 让RBM产生1个样本
        obj = gibbs_sample(obj,times) % Gibbs抽样
        hidden = posterior_sample(obj,visual) % 在给定显层神经元取值的情况下，对隐神经元进行抽样
        hidden = posterior(obj,visual) % 在给定显层神经元取值的情况下，计算隐神经元的激活概率
        visual = likelihood_sample(obj,hidden) % 在给定隐层神经元取值的情况下，对显神经元进行抽样
        visual = likelihood(obj,hidden) % 在给定隐层神经元取值的情况下，计算显神经元的激活概率
        recons = reconstruct(obj,visual) % 在给定显层神经元取值的情况下，计算重建显层神经元的取值 
    end
    
    methods (Access = private)
        [delta_weight,delta_hidden_bias,delta_visual_bias,recon_error] = cd1(obj, train_data) % CD1快速训练算法
        obj = initialize(obj,train_data) % 初始化权值矩阵，隐神经元偏置和显神经元偏置
    end
end

