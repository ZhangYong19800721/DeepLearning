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
        
        obj = showit(obj,data,isgray) % ��ͼ����ʾ�¸���������
    end
end

