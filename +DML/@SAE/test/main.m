% 2017-02-08
% author£ºZhangYong, 24452861@qq.com

clear all;
close all;
format compact;


screen_message = 'Preparing train data set ......'
load('mnist.mat');
train_data = mnist_train_images;
train_label = mnist_train_labels;
minibatch_size = 100;
minibatch_num = 0;

for n = 1:length(train_label)
    idx = mod(n-1,10);
    if idx == train_label(n)
        % do nothing!
    else
        flag = false;
        for k = (n+1):length(train_label)
            if idx == train_label(k)
                swap = train_label(k);
                train_label(k) = train_label(n);
                train_label(n) = swap;
                
                swap = train_data(:,k);
                train_data(:,k) = train_data(:,n);
                train_data(:,n) = swap;
                
                flag = true;
                break;
            end
        end
        
        if flag == false
            minibatch_num = floor((n - 1)/minibatch_size);
            break;
        end
    end
end
train_data = train_data(:,1:(minibatch_size*minibatch_num));
train_data = double(reshape(train_data,784,minibatch_size,minibatch_num)) / 255;
train_label = train_label(1:(minibatch_size*minibatch_num));
screen_message = 'Preparing train data set ...... ok!'

screen_message = 'SAE initialized ......'
sae = DML.SAE([784,4096,2048,1024,512,256,128]);
screen_message = 'SAE initialized ...... ok!'

screen_message = 'SAE is training ......'
learn_rate_min = 1e-10;
learn_rate_max = 1e-1;
max_iteration = 16;
sae = sae.train(train_data,learn_rate_min,learn_rate_max,max_iteration);
screen_message = 'SAE is training ...... ok!'

screen_message = 'SAE is training ......'
save('sae_trained.mat');
screen_message = 'SAE is training ...... ok!'
