function rebuild_data = rebuild_sample(obj,data)
%REBUILD 重建数据
%   
    rebuild_data = obj.decode_sample(obj.encode_sample(data));
end

