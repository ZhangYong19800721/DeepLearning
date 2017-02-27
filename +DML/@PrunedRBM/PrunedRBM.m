classdef PrunedRBM
    %PrunedRBM ��֦��Լ������������
    %   
    
    properties
        num_hidden; % The number of hidden units
        num_visual; % The number of visual units
        weight; % The weight matrix, dimension = (number of hidden units * number of visual units)
        hidden_bias; % The bias for hidden units
        visual_bias; % The bias for visual units
        hidden_unit; % The state of hidden neural
        visual_unit; % The state of visual neural
        mask_weight; % Ȩֵ��mask����
        mask_visual; % �Բ���Ԫ��mask����
        mask_hidden; % ������Ԫ��mask����
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
        obj = initialize(obj,train_data) % ��ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
        obj = initialize_whv(obj,weight,hidden_bias,visual_bias) % ʹ��Ȩֵ������Ԫƫ�á�����Ԫƫ��ֱ�ӳ�ʼ��RBM
        obj = initialize_sgd(obj,directory,filename,minibatch_num,below_layers) % ��ʼ��Ȩֵ��������Ԫƫ�ú�����Ԫƫ��
        obj = initialize_sgd2(obj,train_data,below_layers)
        obj = train(obj,train_data,learn_rate_min,learn_rate_max,max_iteration) % CD1����ѵ���㷨
        obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate,max_iteration,below_layers) % ����ݶ��½�����һ���ܴ�����ݼ��Ͻ���ѧϰ�����ݱ���Ϊ���minibatch
        obj = train_sgd2(obj,train_data,learn_rate_min,learn_rate_max,max_iteration,below_layers)
        [example, obj] = generate(obj) % ��RBM����1������
        obj = gibbs_sample(obj,times) % Gibbs����
        hidden = posterior_sample(obj,visual) % �ڸ����Բ���Ԫȡֵ������£�������Ԫ���г���
        hidden = posterior(obj,visual) % �ڸ����Բ���Ԫȡֵ������£���������Ԫ�ļ������
        visual = likelihood_sample(obj,hidden) % �ڸ���������Ԫȡֵ������£�������Ԫ���г���
        visual = likelihood(obj,hidden) % �ڸ���������Ԫȡֵ������£���������Ԫ�ļ������
        recons = reconstruct(obj,visual) % �ڸ����Բ���Ԫȡֵ������£������ؽ��Բ���Ԫ��ȡֵ 
    end
    
    methods (Access = private)
        [delta_weight,delta_hidden_bias,delta_visual_bias,recon_error] = cd1(obj, train_data) % CD1����ѵ���㷨
    end
end

