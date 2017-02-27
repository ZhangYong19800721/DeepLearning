clear all;
close all;

load('zym_trained_4_negative_layer784x4096x2048x1024x512.mat');

mnist_test_images_negative = 255 - mnist_test_images;
data = zeros(size(mnist_test_images,1),2*size(mnist_test_images,2));
data(:,1:2:size(data,2)) = mnist_test_images;
data(:,2:2:size(data,2)) = mnist_test_images_negative;

data = double(data) / 255;
code = zym.encode(data);
rebuild_data = zym.decode(code);

noise = double(uint8(255*rebuild_data) - uint8(255*data));
noise_power = mean(sum(noise.^2) / size(data,1));
psnr_dB = 10*log10(255^2/noise_power);

for n = 1:400
    imshow([reshape(uint8(255*data(:,n)),28,28)';reshape(uint8(255*rebuild_data(:,n)),28,28)']);
end

distance = zeros(size(data,2),size(data,2));
for n = 1:size(data,2)
    for m = n:size(data,2)
        distance(n,m) = sum((code(:,n) - code(:,m)).^2);
    end
end

distance = distance' + distance;
distance = 1e5 * eye(size(distance)) + distance;
[min_dis, pos] = min(distance);

for n = 1:400
    imshow([reshape(uint8(255*data(:,n)),28,28)';reshape(uint8(255*data(:,pos(n))),28,28)']);
end
