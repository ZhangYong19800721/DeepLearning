classdef ObserverHist
    %OBSERVERHIST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        handle;
        row;
        column;
    end
    
    methods
        function obj = ObserverHist(name,row,column)
            obj.handle = figure('Name',name);
            obj.row = row;
            obj.column = column;
            for n = 1:(row*column)
                subplot(row,column,n);
            end
        end
        
        obj = showit(obj,data,bar,index) %���Ƚ�dataת��Ϊһά���ݣ�Ȼ����hist���ֳ���,���ڣ�row,column����Ӧ����ͼ��
    end
    
end

