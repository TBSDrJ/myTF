import numpy as np
import random
import tensorflow.keras.models as models
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt

((trainX, trainY), (testX, testY)) = datasets.cifar10.load_data()

threshold = 11

net = models.load_model('cifar_net')

img = np.random.rand(1, 32, 32, 3)
start = img.reshape(32, 32, 3)

result1 = 0
result2 = 0
changes = 0
attempts = 0

print(net.predict(img))

while result1 < threshold and result2 < threshold:
    img2 = img.copy()
    for pixelNum in range(50):
        i = random.randrange(32)
        j = random.randrange(32)
        k = random.randrange(3)
        l = (random.randrange(21) - 10) / 200
        if img2[0][i][j][k] + l < 1:
            if img2[0][i][j][k] + l >= 0:
                img2[0][i][j][k] += l
            else:
                img2[0][i][j][k] = 0
        else:
            img2[0][i][j][k] = 0.99999
    result1 = net.predict(img)[0][6]
    result2 = net.predict(img2)[0][6]
    attempts += 1
    if result2 >= result1:
        img = img2
        changes += 1
        if changes % 50 == 0:
            print(changes, attempts, max(result1, result2))

print(net.predict(img))
print(changes, attempts)

img = img.reshape(32, 32, 3)

redsUp = 0
greensUp = 0
bluesUp = 0
startAvgs = []
imgAvgs = []
for i in range(32):
    for j in range(32):
        imgAvgs.append((img[i][j][0] + img[i][j][1] + img[i][j][2]) / 3)
        startAvgs.append((start[i][j][0] + start[i][j][1] + start[i][j][2]) / 3)
        for k in range(3):
            change = img[i][j][k] - start[i][j][k]
            if k == 0:
                if change > 0:
                    redsUp += 1
                elif change < 0:
                    redsUp -= 1
            elif k == 1:
                if change > 0:
                    greensUp += 1
                elif change < 0:
                    greensUp -= 1
            elif k == 2:
                if change > 0:
                    bluesUp += 1
                elif change < 0:
                    bluesUp -= 1
print(redsUp, greensUp, bluesUp)
print("Before:", sum(startAvgs) / len(startAvgs))
print("After:", sum(imgAvgs) / len(imgAvgs))

black = np.zeros((1, 32, 3))
combo = np.concatenate((start, black, img))
plt.imshow(combo)
plt.show()
