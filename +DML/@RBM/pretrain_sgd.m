function obj = pretrain_sgd(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN_SGD 使用随机梯度下降的方式训练RBM，数据以minibatch的方式组织。
%   
    minibatch_num = size(train_data,3);
    
    observer_window_size = minibatch_num; %观察窗口的大小为
    observer_variable_num = 1; %跟踪变量的个数
    ob1 = VISUAL.Observer('reconstruction error',observer_variable_num,observer_window_size,'xxx'); %初始化第1个Observer，用来观察reconstruction error。
    
    % 初始化velocity变量
    velocity_weight      = zeros(size(obj.weight));
    velocity_hidden_bias = zeros(size(obj.hidden_bias));
    velocity_visual_bias = zeros(size(obj.visual_bias));
    
    % 初始化动量倍率为0.5
    momentum = 0.5; 
    
    % flag = false; % 初始化标签变量，该变量表示特定窗口时间长度的滑动平均的reconstruction error是否被初始化
    
    recon_error_list = zeros(1,minibatch_num);
    for minibatch_index = 1:minibatch_num  % 初始化reconstruction error的移动平均值
        minibatch = train_data(:,:,minibatch_index);
        [~, ~, ~, recon_error] = cd1(obj,minibatch);
        recon_error_list(minibatch_index) = recon_error;
    end
    recon_error_average_old = mean(recon_error_list);
    ob1 = ob1.init_data(recon_error_average_old);
    
    learn_rate = learn_rate_max; %初始化学习速度    

    for iteration = 1:max_iteration
        % 取一个minibatch
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
        
        momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
        velocity_weight      = momentum * velocity_weight      + learn_rate * delta_weight;
        velocity_hidden_bias = momentum * velocity_hidden_bias + learn_rate * delta_hidden_bias;
        velocity_visual_bias = momentum * velocity_visual_bias + learn_rate * delta_visual_bias;
        
        obj.weight = obj.weight + velocity_weight;
        obj.hidden_bias = obj.hidden_bias + velocity_hidden_bias;
        obj.visual_bias = obj.visual_bias + velocity_visual_bias;
    end
end

