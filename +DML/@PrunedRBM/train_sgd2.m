function obj = train_sgd2(obj,train_data,learn_rate_min,learn_rate_max,max_iteration,below_layers)
%TRAIN_SGD 使用随机梯度下降的方式训练RBM，数据以minibatch的方式组织。
%   
    minibatch_num = size(train_data,3);
    num_below_layers = length(below_layers);
    observer_window_size = 2 * minibatch_num; %观察窗口的大小为
    observer_variable_num = 1; %跟踪变量的个数
    ob1 = VISUAL.Observer('reconstruction error',observer_variable_num,observer_window_size,'xxx'); %初始化第1个Observer，用来观察reconstruction error。
    
    % 初始化velocity变量
    velocity_weight      = zeros(size(obj.weight));
    velocity_hidden_bias = zeros(size(obj.hidden_bias));
    velocity_visual_bias = zeros(size(obj.visual_bias));
    
    % 初始化动量倍率为0.5
    momentum = 0.5; 
    
    flag = false; % 初始化标签变量，该变量表示特定窗口时间长度的滑动平均的reconstruction error是否被初始化
    recon_error_record = zeros(1,minibatch_num);
    
    learn_rate_current = learn_rate_max; %初始化学习速度
    
    current_minibatch_index = 0; %初始化当前minibatch的序号
    
    for iteration = 1:max_iteration
        if learn_rate_current == learn_rate_min
            break;
        end
        
        % 取一个minibatch
        current_minibatch_index = mod(current_minibatch_index,minibatch_num) + 1;
        minibatch = train_data(:,:,current_minibatch_index);
        
        if num_below_layers ~= 0 %当下面的层数不等于0时，要进行数据映射
            for layer_index = 1:num_below_layers
                minibatch = below_layers(layer_index).posterior(minibatch);
            end
        end
        
        [delta_weight, delta_hidden_bias, delta_visual_bias, recon_error] = cd1(obj,minibatch);
        
        if flag == false % 初始化reconstruction error的移动平均值
            recon_error_record = repmat(recon_error,1,minibatch_num);
            simple_average_old = 10 * recon_error;
            flag = true;
        end
        
        recon_error_record(current_minibatch_index) = recon_error;
        
        if current_minibatch_index == minibatch_num
            simple_average = sum(recon_error_record)/length(recon_error_record);
            if simple_average > simple_average_old
                learn_rate_current = max(0.5*learn_rate_current,learn_rate_min);
            end
            simple_average_old = simple_average;
        end
        
        recon_error_average = sum(recon_error_record)/length(recon_error_record);
        
        titlename = strcat(strcat(strcat('iteration num : ',num2str(iteration)),' / '),num2str(max_iteration)); 
        titlename = strcat(titlename,strcat(' learn rate : ',num2str(learn_rate_current)));
        ob1 = ob1.showit(recon_error_average,titlename);
        
        momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
        velocity_weight      = momentum * velocity_weight      + learn_rate_current * delta_weight;
        velocity_hidden_bias = momentum * velocity_hidden_bias + learn_rate_current * delta_hidden_bias;
        velocity_visual_bias = momentum * velocity_visual_bias + learn_rate_current * delta_visual_bias;
        
        obj.weight = obj.weight + velocity_weight;
        obj.hidden_bias = obj.hidden_bias + velocity_hidden_bias;
        obj.visual_bias = obj.visual_bias + velocity_visual_bias;
    end
end

