function obj = showit(obj,data,bar,index)
%SHOWIT
%   Detailed explanation goes here
    figure(obj.handle);
    subplot(obj.row,obj.column,index);
    data = reshape(data',1,[]);
    hist(data,bar);
end

