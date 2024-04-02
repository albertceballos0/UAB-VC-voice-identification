
img = imread("highway\input\in001051.jpg");
figure(1);
imshow(img)

train = zeros(150, 240, 320, "uint8");
trainGT = zeros(150, 240, 320, "uint8");
test = zeros(150, 240, 320, "uint8");
testGT = zeros(150, 240, 320, "uint8");

for i = 1051:1200
    train(i - 1050, :, :) = rgb2gray(imread(sprintf("highway/input/in%06d.jpg", i))); 
    trainGT(i-1050, :, :) = imread(sprintf("highway/groundtruth/gt%06d.png", i));
end
for i = 1201:1350
    test(i - 1200, :, :) = rgb2gray(imread(sprintf("highway/input/in%06d.jpg", i))); 
    testGT(i-1200, :, :) = imread(sprintf("highway/groundtruth/gt%06d.png", i));
end

trainDouble = double(train);
testDouble = double(test);

mitjana = mean(train);
stdev = std(trainDouble);

mitjanaTest = mean(test);
stdevTest = std(testDouble);

%THRESHOLD ESTATIC

results = zeros(10, 150, 240, 320, "logical");
j = 1;
for i = 20:10:110
    results(j, :, :, :) = ((trainDouble - mitjana) > i) | ((trainDouble - mitjana) < -i);
    j = j+1;
end

figure(2);
for i = 1:10
    subplot(2,5,i)
    resulti = squeeze(results(i,:, :, :));
    imshow(squeeze(resulti(150,  :, :)));
end

figure(3);
subplot(1, 3, 1);
imshow(squeeze(train(150, :, :)));
subplot(1, 3, 2);
imshow(squeeze(results(5, 150, :, :)));
subplot(1, 3, 3);
imshow(squeeze(trainGT(150, :, :)));

precisions_img_e = zeros(1, 150);
precisions_th_e = zeros(1, 10);
recalls_img_e = zeros(1, 150);
recalls_th_e = zeros(1, 10);
f1_scores_img_e = zeros(1, 150);
f1_scores_th_e = zeros(1, 10);

for i = 1:10
    for j = 1:150
        [precisions_img_e(j), recalls_img_e(j), f1_scores_img_e(j)] = compute_scores(squeeze(results(i, j, :, :)), squeeze(trainGT(j, :, :)));
    end
    precisions_th_e(i) = mean(precisions_img_e);
    recalls_th_e(i) = mean(recalls_img_e);
    f1_scores_th_e(i) = mean(f1_scores_img_e);
end

precisions_th_e
recalls_th_e
f1_scores_th_e

figure(8)
plot(f1_scores_th_e);
title("F1-score per threshold estatic");

% COM EL F1_SCORE MAJOR ES A THRESHOLD = 40, UTILITZAREM AQUEST THRESHOLD
% PER AL TEST

resultsTest = zeros(150, 240, 320, "logical");
resultsTest = ((testDouble - mitjanaTest) > 40) | ((testDouble - mitjanaTest) < -40);


precisions_img_e_test = zeros(1, 150);
recalls_img_e_test = zeros(1, 150);
f1_scores_img_e_test = zeros(1, 150);

for j = 1:150
    [precisions_img_e_test(j), recalls_img_e_test(j), f1_scores_img_e_test(j)] = compute_scores(squeeze(resultsTest(j, :, :)), squeeze(testGT(j, :, :)));
end
precisionTest_e = mean(precisions_img_e_test);
recallTest_e = mean(recalls_img_e_test);
f1_scoreTest_e = mean(f1_scores_img_e_test);


precisionTest_e
recallTest_e
f1_scoreTest_e


% THRESHOLD DINAMIC

results_dynamicThreshold = zeros(5, 10, 150, 240, 320, "logical");
trainSenseMitjana = trainDouble - mitjana;
for alpha = 1:5
    index2 = 1;
    for beta = 20:10:110
        results_dynamicThreshold(alpha, index2, :, :, :) = (trainSenseMitjana) > (alpha * stdev + beta);
        index2 = index2 + 1;
    end
    
end

precisions_img_d = zeros(1, 150);
precisions_th_d = zeros(5, 10);
recalls_img_d = zeros(1, 150);
recalls_th_d = zeros(5, 10);
f1_scores_img_d = zeros(1, 150);
f1_scores_th_d = zeros(5, 10);

for alpha = 1:5
    for beta = 1:10
        for j = 1:150
            [precisions_img_d(j), recalls_img_d(j), f1_scores_img_d(j)] = compute_scores(squeeze(results_dynamicThreshold(alpha, beta, j, :, :)), squeeze(trainGT(j, :, :)));
        end
        precisions_th_d(alpha, beta) = mean(precisions_img_d);
        recalls_th_d(alpha, beta) = mean(recalls_img_d);
        f1_scores_th_d(alpha, beta) = mean(f1_scores_img_d);
    end
end

precisions_th_d
recalls_th_d
f1_scores_th_d

figure
plot(1:10, f1_scores_th_d(1, :, :),'g',1:10, f1_scores_th_d(2, :, :),'b',1:10,f1_scores_th_d(3, :, :),'r',1:10,f1_scores_th_d(4, :, :),'m',1:10,f1_scores_th_d(5, :, :),'c');
legend("alpha = 1", "alpha = 2" ,"alpha = 3", "alpha = 4", "alpha = 5")
title("F1-score per threshold dinamic (alpha i beta)")
xlabel("Beta")

% EL MILLOR F1-SCORE RESULTA DE ALPHA = 1, BETA = 1, PER TANT SON ELS
% THRESHOLD QUE UTILITZAREM PER EL TEST

results_dynamicThreshold_test = (testDouble - mitjanaTest) > (1 * stdev + 1);

precisions_img_d_test = zeros(1, 150);
recalls_img_d_test = zeros(1, 150);
f1_scores_img_d_test = zeros(1, 150);

for j = 1:150
   [precisions_img_d(j), recalls_img_d(j), f1_scores_img_d(j)] = compute_scores(squeeze(results_dynamicThreshold_test(j, :, :)), squeeze(trainGT(j, :, :)));   
end



precisionTest_d = mean(precisions_img_d_test);
recallTest_d = mean(recalls_img_d_test);
f1_scoreTest_d = mean(f1_scores_img_d_test);


precisionTest_d
recallTest_d
f1_scoreTest_d






function [precision, recall, f1_score] = compute_scores(test_image, ground_truth_image)
    % Convert images to logical
    test_image = logical(test_image);
    ground_truth_image = logical(ground_truth_image);

    % Compute true positives, false positives, and false negatives
    TP = sum(test_image(:) & ground_truth_image(:));
    FP = sum(test_image(:) & ~ground_truth_image(:));
    FN = sum(~test_image(:) & ground_truth_image(:));

    % Compute precision and recall
    if (TP + FP) == 0
        precision = 0;
    else
        precision = TP / (TP + FP);
    end
    if (TP + FN) == 0
        recall = 0;
    else
        recall = TP / (TP + FN);
    end

    % Compute F1-score
    if (precision + recall) == 0
        f1_score = 0;
    else
        f1_score = 2 * (precision * recall) / (precision + recall);
    end
end








