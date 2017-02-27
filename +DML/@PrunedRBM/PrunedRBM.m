classdef PrunedRBM
    %PrunedRBM 剪枝的约束玻尔兹曼机
    %   
    
    properties
        num_hidden; % The number of hidden units
        num_visual; % The number of visual units
        weight; % The weight matrix, dimension = (number of hidden units * number of visual units)
        hidden_bias; % The bias for hidden units
        visual_bias; % The bias for visual units
        hidden_unit; % The state of hidden neural
        visual_unit; % The state of visual neural
        mask_weight; % 权值的mask矩阵
        mask_visual; % 显层神经元的mask矩阵
        mask_hidden; % 隐层神经元的mask矩阵
    end
    
    methods
        function obj = PrunedRBM(num_hidden,num_visual) % The constructor functions
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.mask_weight = ones(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
            obj.hidden_unit = zeros(obj.num_hidden,1);
            obj.visual_unit = zeros(obj.num_visual,1);
            obj.mask_visual = ones(obj.num_visual,1);
            obj.mask_hidden = ones(obj.num_hidden,1);
        end
    end
    
    methods
        obj = initialize(obj,train_data) % 初始化权值矩阵，隐神经元偏置和显神经元偏置
        obj = initialize_whv(obj,weight,hidden_bias,visual_bias) % 使用权值、隐神经元偏置、显神经元偏置直接初始化RBM
        obj = initialize_sgd(obj,directory,filename,minibatch_num,below_layers) % 初始化权值矩阵，隐神经元偏置和显神经元偏置
        obj = initialize_sgd2(obj,train_data,below_layers)
        obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % CD1快速训练算法
        obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate,max_iteration,below_layers) % 随机梯度下降，在一个很大的数据集上进行学习，数据被分为多个minibatch
        obj = train_sgd2(obj,train_data,learn_rate_min,learn_rate_max,max_iteration,below_layers)
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
    end
end

