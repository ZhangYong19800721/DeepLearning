function rebuild_data = rebuild_sample(obj,data)
%REBUILD �ؽ�����
%   
    rebuild_data = obj.decode_sample(obj.encode_sample(data));
end

