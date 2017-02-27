function recons = reconstruct( obj, visual )
%RECONSTRUCT 在给定显层神经元取值的情况下，计算重建显层神经元的取值
%   obj：RBM（约束玻尔兹曼机）
%   visual: 显层神经元的取值，每个值的范围在[0,1]之间，不要求必须是二元取值
%   recons：输出的重建值，每个值的范围在[0,1]之间，不要求必须是二元取值

    hidden_unit = obj.posterior_sample(visual);
    recons = obj.likelihood(hidden_unit);
end

