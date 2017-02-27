function obj = wake_sleep(obj,train_data,learn_rate_min,learn_rate_max,max_iteration)
%WAKE_SLEEP 使用wake-sleep算法进行训练
%
    learn_rate = learn_rate_max; % 将学习速度初始化为最大学习速度
    [visual_num,minibatch_size,minibatch_num] = size(train_data);
    
    observer_window_size = minibatch_num; %观察窗口的大小为
    observer_variable_num = 1; %跟踪变量的个数
    ob = VISUAL.Observer('rebuild_error_ave',observer_variable_num,observer_window_size,'xxx'); %初始化第2个Observer，用来观察reconstruction error average。
    
    minibatch = reshape(train_data,visual_num,minibatch_size * minibatch_num);
    rebuild_data = obj.rebuild(minibatch);
    rebuild_error = sum(sum(abs(rebuild_data - minibatch))) / size(minibatch,2);
    rebuild_error_average_old = rebuild_error;
    rebuild_error_list = rebuild_error_average_old * ones(1,observer_window_size);
    ob = ob.init_data(rebuild_error_average_old);
    
    % 初始化velocity变量
    num_of_layers = length(obj.decoder_layers);
    
    velocity_wake.top_bias = zeros(size(obj.decoder_layers(num_of_layers).rbm.hidden_bias));
    for n = 1:num_of_layers
        velocity_wake(n).weight = zeros(size(obj.decoder_layers(n).rbm.weight));
        velocity_wake(n).visual_bias = zeros(size(obj.decoder_layers(n).rbm.visual_bias));
    end
    
    for n = 1:num_of_layers
        velocity_sleep(n).weight = zeros(size(obj.encoder_layers(n).rbm.weight));
        velocity_sleep(n).hidden_bias = zeros(size(obj.encoder_layers(n).rbm.hidden_bias));
    end
    
    % 初始化动量倍率为0.5
    momentum = 0.5;
    for it = 1:max_iteration 
        momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
        minibatch_index = mod(it,minibatch_num) + 1;
        minibatch = train_data(:,:,minibatch_index);
        
        delta_wake = obj.wake(minibatch); % wake 阶段
        velocity_wake(1).top_bias = momentum * velocity_wake(1).top_bias + learn_rate * delta_wake(1).top_bias;
        obj.decoder_layers(num_of_layers).rbm.hidden_bias = obj.decoder_layers(num_of_layers).rbm.hidden_bias + velocity_wake(1).top_bias;
        for n = 1:num_of_layers
            velocity_wake(n).weight = momentum * velocity_wake(n).weight + learn_rate * delta_wake(n).weight;
            obj.decoder_layers(n).rbm.weight = obj.decoder_layers(n).rbm.weight + velocity_wake(n).weight;
            velocity_wake(n).visual_bias = momentum * velocity_wake(n).visual_bias + learn_rate * delta_wake(n).visual_bias;
            obj.decoder_layers(n).rbm.visual_bias = obj.decoder_layers(n).rbm.visual_bias + velocity_wake(n).visual_bias;
        end
        
        delta_sleep = obj.sleep(minibatch); % sleep 阶段
        for n = 1:num_of_layers
            velocity_sleep(n).weight = momentum * velocity_sleep(n).weight + learn_rate * delta_sleep(n).weight;
            obj.encoder_layers(n).rbm.weight = obj.encoder_layers(n).rbm.weight + velocity_sleep(n).weight;
            velocity_sleep(n).hidden_bias = momentum * velocity_sleep(n).hidden_bias + learn_rate * delta_sleep(n).hidden_bias;
            obj.encoder_layers(n).rbm.hidden_bias = obj.encoder_layers(n).rbm.hidden_bias + velocity_sleep(n).hidden_bias;
        end
        
        % 计算重建误差
        rebuild_data = obj.rebuild(minibatch);
        rebuild_error = sum(sum(abs(rebuild_data - minibatch))) / size(minibatch,2);
        rebuild_error_list(mod(it,observer_window_size)+1) = rebuild_error;
        rebuild_error_average = mean(rebuild_error_list);
        
        if mod(it,observer_window_size) == 0
            if rebuild_error_average > rebuild_error_average_old % 当经过N次迭代的平均重建误差不下降时
                learn_rate = learn_rate / 2; % 就缩减学习速度
                if learn_rate < learn_rate_min % 当学习速度小于最小学习速度时，退出
                    break;
                end
            end
            rebuild_error_average_old = rebuild_error_average;
        end
        
        % 画图
        titlename = strcat(strcat(strcat('wake learning - step : ',num2str(it)),'/ '),num2str(max_iteration));
        titlename = strcat(titlename,strcat(';  rate : ',num2str(learn_rate)));
        ob = ob.showit(rebuild_error_average,titlename);
    end
end

