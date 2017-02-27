function obj = showit(obj,data,isgray)
%SHOWIT
%
    figure(obj.handle);
    min_value = min(min(data));
    max_value = max(max(data));
    data = (data - min_value) / (max_value - min_value) * 64;
    if isgray
        colormap(gray);
    end
    image(data);
end

