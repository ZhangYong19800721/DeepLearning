function output = Forward(obj,input)
%FORWARD �ڵ����֪���У�����������Ԫ��״̬���������Ԫ��״̬
%
    output = sigmoid(obj.weight * input + obj.bias);
end

