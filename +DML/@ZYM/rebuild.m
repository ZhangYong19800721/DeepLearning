function rebuild_data = rebuild(obj,data)
%REBUILD �ؽ�����
%   
    rebuild_data = obj.decode(obj.encode(data));
end

