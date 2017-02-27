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
        
        obj = showit(obj,data,bar,index) %首先将data转换为一维数据，然后用hist表现出来,画在（row,column）对应的子图中
    end
    
end

