classdef DBN
%DBN Deep Belief Nets,深度信度网络   
%
    properties
        rbm_softmax;
        rbm_layer_1;
        rbm_layer_2;
    end
    
    methods
        function obj = DBN(v0,h1,h2,h3,num_softmax)
            obj.rbm_layer_1 = DML.RBM(h1,v0);
            obj.rbm_layer_2 = DML.RBM(h2,h1);
            obj.rbm_softmax = DML.RBM_SOFTMAX(h3,h2 + num_softmax,num_softmax);
        end
    end
    
    methods
        obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate_min,learn_rate_max,max_iteration) % 使用CD1快速算法，逐层训练DBN
        type = discriminate(obj,data) % 给定显神经元的取值，识别其类别 
        data = recall(obj,type) % 给定softmax神经元的取值，回忆数据
    end
end

