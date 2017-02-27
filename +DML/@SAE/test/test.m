% 2017-02-08
% author£ºZhangYong, 24452861@qq.com

clear all;
close all;

load('asea_trained.mat');
load('mnist.mat');

format compact

print_screen = 'test ......'

datas = double(mnist_test_images) ./ 255;

% for n = 6:-1:1
%     codes = ase.encode(datas,n);
%     eatas = ase.decode(codes,n);
%     image = double(uint8(255*datas));
%     recon = double(uint8(255*eatas));
%     psnr = 255^2 ./ (sum((image-recon).^2,1) ./ 784);
%     result(n).ave_psnr = mean(psnr);
%     result(n).ave_psnr_dB = 10 * log10(result(n).ave_psnr);
% end

psnr = zeros(1,10000);
for n = 1:10000
    data = datas(:,n);
    code = ase.encode_vbr(data);
    eata = ase.decode_vbr(code);
    noise = double(uint8(255*data)) - double(uint8(255*eata));
    psnr(n) =  255^2 ./ (sum(noise.^2)./784);
end

ave_psnr_vbr = mean(psnr);
ave_psnr_vbr = 10*log10(ave_psnr_vbr)




