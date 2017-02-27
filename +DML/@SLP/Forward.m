function output = Forward(obj,input)
%FORWARD 在单层感知器中，根据输入神经元的状态计算输出神经元的状态
%
    output = sigmoid(obj.weight * input + obj.bias);
end

