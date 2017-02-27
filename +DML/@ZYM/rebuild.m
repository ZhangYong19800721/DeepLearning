function rebuild_data = rebuild(obj,data)
%REBUILD 重建数据
%   
    rebuild_data = obj.decode(obj.encode(data));
end

