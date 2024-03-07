import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
imgs = [file for file in os.listdir("./highway/input")]
imgsGT = [file for file in os.listdir("./highway/groundtruth")]

train = []
trainGT = [] 
for i in range(1051,1350):   
    train.append(cv2.cvtColor(cv2.imread('./highway/input/' +imgs[i]), cv2.COLOR_BGR2GRAY))
    trainGT.append(cv2.imread('./highway/groundtruth/' +imgsGT[i]))

train_imgs = np.array(train)
train_imgs = train_imgs.astype(int)

train_imgsGT = np.array(trainGT)

mean = np.mean(train_imgs, 0)
std = np.std(train_imgs, 0)

plt.imshow(std, cmap = "gray");
#plt.show()

results = (train_imgs - mean ) > 70


plt.imshow(train_imgs[3], cmap='gray')
plt.show()


plt.imshow(results[3], cmap='gray')
plt.show()

plt.imshow(train_imgsGT[3], cmap='gray')

plt.show()