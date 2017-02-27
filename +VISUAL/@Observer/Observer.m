classdef Observer
    %OBSERVER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        handle;
        window;
        legend;
        index;
        list;
        num;
    end
    
    methods
        function obj = Observer(name,num,window,legend)
            obj.handle = figure('Name',name); grid on;
            obj.window = window;
            obj.legend = legend;
            obj.num = num;
            obj.list = zeros(obj.num,obj.window);
            obj.index = 0;
        end
        
        obj = showit(obj,data,title_line) % ��ͼ����ʾ�¸���������
        obj = init_data(obj,data) % ��ʼ��ͼ�еĳ�ʼ����
    end
end

