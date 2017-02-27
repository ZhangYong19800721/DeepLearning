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
    
    % flag = false; % ��ʼ����ǩ�������ñ�����ʾ�ض�����ʱ�䳤�ȵĻ���ƽ����reconstruction error�Ƿ񱻳�ʼ��
    
    recon_error_list = zeros(1,minibatch_num);
    for minibatch_index = 1:minibatch_num  % ��ʼ��reconstruction error���ƶ�ƽ��ֵ
        minibatch = train_data(:,:,minibatch_index);
        [~, ~, ~, recon_error] = cd1(obj,minibatch);
        recon_error_list(minibatch_index) = recon_error;
    end
    recon_error_average_old = mean(recon_error_list);
    ob1 = ob1.init_data(recon_error_average_old);
    
    learn_rate = learn_rate_max; %��ʼ��ѧϰ�ٶ�    

    for iteration = 1:max_iteration
        % ȡһ��minibatch
        minibatch_index = mod(minibatch_index,minibatch_num) + 1;
        minibatch = train_data(:,:,minibatch_index);
        
        [delta_weight, delta_hidden_bias, delta_visual_bias, recon_error] = cd1(obj,minibatch);
        recon_error_list(minibatch_index) = recon_error;
        
        if minibatch_index == minibatch_num
            recon_error_average = mean(recon_error_list);
            if recon_error_average > recon_error_average_old
                learn_rate = learn_rate / 2;
                if learn_rate < learn_rate_min
                    break;
                end
            end
            recon_error_average_old = recon_error_average;
        end
        
        recon_error_average = mean(recon_error_list);
        
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

