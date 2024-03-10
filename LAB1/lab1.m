
%images = zeros(300, 240, 320, 3)


img = imread("highway\input\in001051.jpg");
figure(1);
imshow(img)

train = zeros(300, 240, 320, "uint8");
trainGT = zeros(300, 240, 320, "uint8");

for i = 1051:1350
    train(i - 1050, :, :) = rgb2gray(imread(sprintf("highway/input/in%06d.jpg", i))); 
    trainGT(i-1050, :, :) = imread(sprintf("highway/groundtruth/gt%06d.png", i));
end

trainDouble = double(train);

mitjana = mean(train);
stdev = std(trainDouble);

results = (trainDouble - mitjana) > 70;
figure(2);
imshow(squeeze(results(1, :, :)));


