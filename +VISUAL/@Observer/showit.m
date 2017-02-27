function obj = showit(obj,data,title_line)
    obj.index = mod(obj.index,obj.window);
    obj.list(:,obj.index+1) = data;
    
    figure(obj.handle);
    
    switch obj.num
        case 1
            plot(1:obj.window,obj.list(1,:));
        case 2
            plot(1:obj.window,obj.list(1,:),1:obj.window,obj.list(2,:));
        case 3
            plot(1:obj.window,obj.list(1,:),1:obj.window,obj.list(2,:),1:obj.window,obj.list(3,:));
        case 4
            plot(1:obj.window,obj.list(1,:),1:obj.window,obj.list(2,:),1:obj.window,obj.list(3,:),1:obj.window,obj.list(4,:));
    end
    grid on;
    title(title_line);
    xlim([0 obj.window]);
    obj.index = obj.index + 1;
    drawnow
end

