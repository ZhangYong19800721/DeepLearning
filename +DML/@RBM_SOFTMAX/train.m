function obj = train( obj,train_data,learn_rate )
%train 
%   Detailed explanation goes here
    observer_window_size = 542; %观察窗口的大小为
    observer_variable_num = 3; %跟踪变量的个数
    ob1 = VISUAL.Observer('reconstruction error',observer_variable_num,observer_window_size,'xxx');
    
    % 初始化velocity变量
    velocity_weight = zeros(size(obj.weight));
    velocity_hidden_bias = zeros(size(obj.hidden_bias));
    velocity_visual_bias = zeros(size(obj.visual_bias));
    
    % 初始化动量倍率为0.5
    momentum = 0.5; 
    
    flag = false; % 初始化标签变量，该变量表示特定窗口时间长度的滑动平均的reconstruction error是否被初始化
    recon_error_average = (1:observer_variable_num)' - observer_variable_num;
    
    learn_rate_max = learn_rate; %初始化最大学习速度
    
    while true 
        if recon_error_average(2) > recon_error_average(3)
            %当500期均线超过2500期均线时停止迭代
            break;
        end
        
        [delta_weight delta_hidden_bias delta_visual_bias recon_error] = cd1(obj,train_data);
        
        if flag == false
            recon_error_average = recon_error * [1.0 1.1 1.2];
            flag = true;
        end

        alfa =  100;  recon_error_average(1) = (alfa-1)/alfa * recon_error_average(1)  + 1/alfa * recon_error;
        alfa =  500;  recon_error_average(2) = (alfa-1)/alfa * recon_error_average(2)  + 1/alfa * recon_error;
        alfa = 2500;  recon_error_average(3) = (alfa-1)/alfa * recon_error_average(3)  + 1/alfa * recon_error;
        
        ob1 = ob1.showit(recon_error_average);
        
        if recon_error_average(1) < recon_error_average(2)
            learn_rate = learn_rate_max;
        elseif recon_error_average(1) < recon_error_average(3)
            learn_rate = learn_rate_max / 2;
        else
            learn_rate = learn_rate_max / 4;
        end
        
        momentum = min([momentum * 1.01,0.9]); % 动量倍率最大为0.9，初始值为0.5，大约迭代60步之后动量倍率达到0.9。
        velocity_weight      = momentum * velocity_weight      + learn_rate * delta_weight;
        velocity_hidden_bias = momentum * velocity_hidden_bias + learn_rate * delta_hidden_bias;
        velocity_visual_bias = momentum * velocity_visual_bias + learn_rate * delta_visual_bias;
        
        obj.weight = obj.weight + velocity_weight;
        obj.hidden_bias = obj.hidden_bias + velocity_hidden_bias;
        obj.visual_bias = obj.visual_bias + velocity_visual_bias;
    end
end

