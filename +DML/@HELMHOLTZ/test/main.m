% 2017-02-15
% author£ºZhangYong, 24452861@qq.com

clear all;
close all;
format compact;

screen_message = 'Preparing train data set ......'
prepare_mnist_train_data;
screen_message = 'Preparing train data set ...... ok!'

screen_message = 'HELMHOLTZ initialized ......'
helmholtz = DML.HELMHOLTZ([784,500,250]);
screen_message = 'HELMHOLTZ initialized ...... ok!'

screen_message = 'HELMHOLTZ is training ......'
learn_rate_min = 1e-10;
learn_rate_max = 1e-1;
max_iteration = 1e6;
helmholtz = helmholtz.train(train_data,learn_rate_min,learn_rate_max,max_iteration);
screen_message = 'HELMHOLTZ is training ...... ok!'

screen_message = 'HELMHOLTZ is saving ......'
save('helmholtz_trained.mat');
screen_message = 'HELMHOLTZ is saving ...... ok!'
