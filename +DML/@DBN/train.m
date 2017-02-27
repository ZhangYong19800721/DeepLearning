function obj = train(obj,directory,filename,minibatch_num,learn_rate_min,learn_rate_max,max_iteration)
%TRAIN ѵ��DBN
%   
    minibatch_size = 100;
    train_data = zeros(obj.rbm_layer_1.num_visual,minibatch_size,minibatch_num);
    
    for minibatch_index = 1:minibatch_num
        minibatch_filename = strcat(directory,strcat(filename,strcat('_',strcat(num2str(minibatch_index),'.txt'))));
        train_data(:,:,minibatch_index) = importdata(minibatch_filename);
    end
    
    screen_message = 'Initialize the first layer RBM ......'
    obj.rbm_layer_1 = obj.rbm_layer_1.initialize_sgd2(train_data,[]); % ��ʼ����1���RBM
    
    screen_message = 'Train the first layer RBM ......';
    obj.rbm_layer_1 = obj.rbm_layer_1.train_sgd2(train_data,learn_rate_min,learn_rate_max,max_iteration,[]); % ѵ����1���RBM
   
    for minibatch_index = 1:minibatch_num %��ѵ������ӳ�䵽��һ��
        train_data(:,:,minibatch_index) = obj.rbm_layer_1.posterior(train_data(:,:,minibatch_index));
    end
    
    screen_message = 'Initialize the second layer RBM ......'
    obj.rbm_layer_2 = obj.rbm_layer_2.initialize_sgd2(train_data,[]); % ��ʼ����2���RBM
    
    screen_message = 'Train the second layer RBM ......'
    obj.rbm_layer_2 = obj.rbm_layer_2.train_sgd2(train_data,learn_rate_min,learn_rate_max,max_iteration,[]); % ѵ����2���RBM
    
    for minibatch_index = 1:minibatch_num %��ѵ������ӳ�䵽��һ��
        train_data(:,:,minibatch_index) = obj.rbm_layer_2.posterior(train_data(:,:,minibatch_index));
    end
    
    screen_message = 'Initialize the softmax layer RBM ......'
    obj.rbm_softmax = obj.rbm_softmax.initialize_sgd2(train_data,[]); % ��ʼ����3���RBM
    
    screen_message = 'Train the softmax layer RBM ......'
    obj.rbm_softmax = obj.rbm_softmax.train_sgd2(train_data,learn_rate_min,learn_rate_max,max_iteration,[]); % ѵ����3���RBM
end

