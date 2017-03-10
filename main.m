clear all;
close all;

[train_data] = import_mnist('mnist.mat');

rbm = DML.RBM(500,784);
rbm = rbm.initialize_sgd(train_data);
rbm = rbm.pretrain_sgd(train_data,1e-10,0.1,1e6);


