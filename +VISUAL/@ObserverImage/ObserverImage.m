classdef ObserverImage
    %OBSERVERIMAGE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        handle;
    end
    
    methods
        function obj = ObserverImage(name)
            obj.handle = figure('Name',name);
        end
        
        obj = showit(obj,data,isgray) % 在图中显示新给出的数据
    end
end

