function obj = train_sgd(obj,directory,filename,minibatch_num,learn_rate,max_iteration,below_layers)
%TRAIN_SGD ʹ������ݶ��½��ķ�ʽѵ��RBM��������minibatch�ķ�ʽ��֯��
%   
    num_below_layers = length(below_layers);
    observer_window_size = 2 * minibatch_num; %�۲촰�ڵĴ�СΪ
    observer_variable_num = 3; %���ٱ����ĸ���
    ob1 = VISUAL.Observer('reconstruction error',observer_variable_num,observer_window_size,'xxx'); %��ʼ����1��Observer�������۲�reconstruction error��
    
    % ��ʼ��velocity����
    velocity_weight      = zeros(size(obj.weight));
    velocity_hidden_bias = zeros(size(obj.hidden_bias));
    velocity_visual_bias = zeros(size(obj.visual_bias));
    
    % ��ʼ����������Ϊ0.5
    momentum = 0.5; 
    
    flag = false; % ��ʼ����ǩ�������ñ�����ʾ�ض�����ʱ�䳤�ȵĻ���ƽ����reconstruction error�Ƿ񱻳�ʼ��
    recon_error_average = (1:observer_variable_num)' - observer_variable_num;
    
    learn_rate_max = learn_rate; %��ʼ�����ѧϰ�ٶ�
    
    current_minibatch_index = 0; %��ʼ����ǰminibatch�����
    
    for iteration = 1:max_iteration
        if recon_error_average(2) > recon_error_average(3)  
            break; %��500�ھ��߳���2500�ھ���ʱֹͣ����
        end
        
        % ���ļ��ж���һ��minibatch
        current_minibatch_index = mod(current_minibatch_index,minibatch_num) + 1;
        minibatch_filename = strcat(directory,strcat(filename,strcat('_',strcat(num2str(current_minibatch_index),'.txt'))));
        minibatch = importdata(minibatch_filename);
        
        if num_below_layers ~= 0 %������Ĳ���������0ʱ��Ҫ��������ӳ��
            for layer_index = 1:num_below_layers
                minibatch = below_layers(layer_index).posterior(minibatch);
            end
        end
        
        [delta_weight, delta_hidden_bias, delta_visual_bias, recon_error] = cd1(obj,minibatch);
        
        if flag == false % ��ʼ��reconstruction error���ƶ�ƽ��ֵ
            recon_error_average = recon_error * [1.0 1.1 1.2];
            flag = true;
        end

        alfa =  100;  recon_error_average(1) = (alfa-1)/alfa * recon_error_average(1)  + 1/alfa * recon_error;
        alfa =  500;  recon_error_average(2) = (alfa-1)/alfa * recon_error_average(2)  + 1/alfa * recon_error;
        alfa = 2500;  recon_error_average(3) = (alfa-1)/alfa * recon_error_average(3)  + 1/alfa * recon_error;
        
        titlename = strcat(strcat(strcat('iteration num : ',num2str(iteration)),' / '),num2str(max_iteration)); 
        ob1 = ob1.showit(recon_error_average,titlename);
        
        learn_rate = learn_rate_max;
%         if recon_error_average(1) < recon_error_average(2)
%             learn_rate = learn_rate_max;
%         elseif recon_error_average(1) < recon_error_average(3)
%             learn_rate = learn_rate_max / 2;
%         else
%             learn_rate = learn_rate_max / 4;
%         end
        
        momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
        velocity_weight      = momentum * velocity_weight      + learn_rate * delta_weight;
        velocity_hidden_bias = momentum * velocity_hidden_bias + learn_rate * delta_hidden_bias;
        velocity_visual_bias = momentum * velocity_visual_bias + learn_rate * delta_visual_bias;
        
        obj.weight = obj.weight + velocity_weight;
        obj.hidden_bias = obj.hidden_bias + velocity_hidden_bias;
        obj.visual_bias = obj.visual_bias + velocity_visual_bias;
    end
end

