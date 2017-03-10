classdef RBM_SOFTMAX
    %RBM Restricted Boltzmann Machine with soft max units
    %   This is the definition of Restricted Boltzmann Machine
    
    properties
        num_hidden; % The number of hidden units
        num_visual; % The number of visual units
        num_softmax; % The number of softmax units
        weight; % The weight matrix, dimension = (number of hidden units * (number of visual units + num of softmax units))
        hidden_bias; % The bias for hidden units
        visual_bias; % The bias for visual units
    end
    
    methods
        function obj = RBM_SOFTMAX(num_hidden,num_visual,num_softmax) % The constructor functions
            obj.num_hidden = num_hidden;
            obj.num_visual = num_visual;
            obj.num_softmax = num_softmax;
            obj.weight = zeros(obj.num_hidden,obj.num_visual);
            obj.hidden_bias = zeros(obj.num_hidden,1);
            obj.visual_bias = zeros(obj.num_visual,1);
        end
    end
    
    methods
        obj = initialize(obj,train_data) % 初始化权值矩阵，隐神经元偏置和显神经元偏置
        obj = initialize_sgd(obj,directory,filename,minibatch_num,below_layers) % 初始化权值矩阵，隐神经元偏置和显神经元偏置
        obj = initialize_sgd2(obj,train_data,below_layers)
        obj = train(obj,train_data,learn_rate) % 用CD1方法训练RBM.
        obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate,max_iteration,below_layers) % 随机梯度下降，在一个很大的数据集上进行学习，数据被分为多个minibatch
        obj = train_sgd2(obj,train_data,learn_rate_min,learn_rate_max,max_iteration,below_layers)
        type = discriminate(obj,data) % 给定显神经元的取值，识别其类别 
        data = recall(obj,type) % 给定softmax神经元的取值，回忆数据
    end
    
    methods (Access = private)
        [delta_weight,delta_hidden_bias,delta_visual_bias,recon_error] = cd1(obj, train_data) % CD1快速训练法
    end
end

