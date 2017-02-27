function obj = train( obj,train_data,learn_rate_min,learn_rate_max,max_iteration )
%TRAIN ʹ��CD1�����㷨ѵ��RBM
%   
    recon_error_average_size = 100; % �ؽ����ƽ�����ڴ�С
    observer_window_size = recon_error_average_size; %�۲촰�ڵĴ�С
    observer_variable_num = 1; %���ٱ����ĸ���
    ob1 = VISUAL.Observer('reconstruction error',observer_variable_num,observer_window_size,'xxx');
    
    velocity_weight = zeros(size(obj.weight));
    velocity_hidden_bias = zeros(size(obj.hidden_bias));
    velocity_visual_bias = zeros(size(obj.visual_bias));
    momentum = 0.5; % ��ʼ����������Ϊ0.5
    flag = false; % ��ʼ����ǩ�������ñ�����ʾ�ض�����ʱ�䳤�ȵĻ���ƽ����reconstruction error�Ƿ񱻳�ʼ��
    learn_rate_current = learn_rate_max; %��ʼ��ѧϰ�ٶ�
    
    for iteration = 1:max_iteration 
        if learn_rate_current == learn_rate_min
            % ��ѧϰ�ٶ��½�����Сѧϰ�ٶ�ʱ��ֹͣ����
            break;
        end
        
        [delta_weight, delta_hidden_bias, delta_visual_bias, recon_error] = cd1(obj,train_data);
        
        if flag == false
            recon_error_record = repmat(recon_error,1,recon_error_average_size);
            simple_average_old = 10 * recon_error;
            flag = true;
        end
        
        recon_error_record(mod(iteration,recon_error_average_size) + 1) = recon_error;
        simple_average = sum(recon_error_record) / length(recon_error_record); 
        
        if mod(iteration,observer_window_size) == 0
            if simple_average > simple_average_old
                learn_rate_current = max(0.5 * learn_rate_current,learn_rate_min);
            end
            simple_average_old = simple_average;
        end
        
        titlename = strcat(strcat(strcat('iteration num : ',num2str(iteration)),' / '),num2str(max_iteration)); 
        titlename = strcat(titlename,strcat(' learn rate : ',num2str(learn_rate_current)));
        ob1 = ob1.showit(simple_average,titlename);
       
        momentum = min([momentum * 1.01,0.9]); % �����������Ϊ0.9����ʼֵΪ0.5����Լ����60��֮�������ʴﵽ0.9��
        velocity_weight      = momentum * velocity_weight      + learn_rate_current * delta_weight;
        velocity_hidden_bias = momentum * velocity_hidden_bias + learn_rate_current * delta_hidden_bias;
        velocity_visual_bias = momentum * velocity_visual_bias + learn_rate_current * delta_visual_bias;
        
        obj.weight = obj.weight + velocity_weight.*obj.mask_weight;
        obj.hidden_bias = obj.hidden_bias + velocity_hidden_bias.*obj.mask_hidden;
        obj.visual_bias = obj.visual_bias + velocity_visual_bias.*obj.mask_visual;
    end
end

