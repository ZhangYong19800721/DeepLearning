function obj = pretrain_sgd(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN_SGD ʹ������ݶ��½��ķ�ʽѵ��RBM��������minibatch�ķ�ʽ��֯��
%   
    minibatch_num = size(train_data,3);
    
    observer_window_size = minibatch_num; %�۲촰�ڵĴ�СΪ
    observer_variable_num = 1; %���ٱ����ĸ���
    ob1 = VISUAL.Observer('reconstruction error',observer_variable_num,observer_window_size,'xxx'); %��ʼ����1��Observer�������۲�reconstruction error��
    
    % ��ʼ��velocity����
    velocity_weight      = zeros(size(obj.weight));
    velocity_hidden_bias = zeros(size(obj.hidden_bias));
    velocity_visual_bias = zeros(size(obj.visual_bias));
    
    % ��ʼ����������Ϊ0.5
    momentum = 0.5; 
    
    recon_error = zeros(1,minibatch_num);
    for minibatch_index = 1:minibatch_num  % ��ʼ��reconstruction error���ƶ�ƽ��ֵ
        minibatch = train_data(:,:,minibatch_index);
        [~, ~, ~, recon_error(minibatch_index)] = cd1(obj,minibatch);
    end
    recon_error_average_old = mean(recon_error);
    ob1 = ob1.init_data(recon_error_average_old);
    
    learn_rate = learn_rate_max; %��ʼ��ѧϰ�ٶ�    

    for iteration = 1:max_iteration
        disp(strcat('iteration =  ',num2str(iteration)))
        delta_weight = zeros(size(obj.weight));
        delta_hidden_bias = zeros(size(obj.hidden_bias));
        delta_visual_bias = zeros(size(obj.visual_bias));
        
        parfor minibatch_index = 1:minibatch_num
            minibatch = train_data(:,:,minibatch_index);
            [d_weight, d_hidden_bias, d_visual_bias, recon_error(minibatch_index)] = cd1(obj,minibatch);
            delta_weight = delta_weight + d_weight;
            delta_hidden_bias = delta_hidden_bias + d_hidden_bias;
            delta_visual_bias = delta_visual_bias + d_visual_bias;
        end

        recon_error_average = mean(recon_error);
        if recon_error_average > recon_error_average_old && iteration > 100
            learn_rate = learn_rate / 10;
            if learn_rate < learn_rate_min
                break;
            end
        end
        recon_error_average_old = recon_error_average;
   
        titlename = strcat(strcat(strcat('iteration num : ',num2str(iteration)),' / '),num2str(max_iteration)); 
        titlename = strcat(titlename,strcat(' learn rate : ',num2str(learn_rate)));
        ob1 = ob1.showit(recon_error_average,titlename);
        
        momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
        velocity_weight      = momentum * velocity_weight      + learn_rate * delta_weight;
        velocity_hidden_bias = momentum * velocity_hidden_bias + learn_rate * delta_hidden_bias;
        velocity_visual_bias = momentum * velocity_visual_bias + learn_rate * delta_visual_bias;
        
        obj.weight = obj.weight + velocity_weight;
        obj.hidden_bias = obj.hidden_bias + velocity_hidden_bias;
        obj.visual_bias = obj.visual_bias + velocity_visual_bias;
    end
end

