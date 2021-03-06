load('./MNIST/mnist.mat');
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
negative_train_data = 255 - train_data;
new_train_data = zeros(size(train_data,1),2*size(train_data,2));
new_train_data(:,1:2:size(new_train_data,2)) = train_data;
new_train_data(:,2:2:size(new_train_data,2)) = negative_train_data;
new_train_data = double(reshape(new_train_data,784,2*minibatch_size,minibatch_num)) / 255;
% train_label = train_label(1:(minibatch_size*minibatch_num));
train_data = new_train_data;