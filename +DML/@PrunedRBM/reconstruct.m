function recons = reconstruct( obj, visual )
%RECONSTRUCT �ڸ����Բ���Ԫȡֵ������£������ؽ��Բ���Ԫ��ȡֵ
%   obj��RBM��Լ��������������
%   visual: �Բ���Ԫ��ȡֵ��ÿ��ֵ�ķ�Χ��[0,1]֮�䣬��Ҫ������Ƕ�Ԫȡֵ
%   recons��������ؽ�ֵ��ÿ��ֵ�ķ�Χ��[0,1]֮�䣬��Ҫ������Ƕ�Ԫȡֵ

    hidden_unit = obj.posterior_sample(visual);
    recons = obj.likelihood(hidden_unit);
end

