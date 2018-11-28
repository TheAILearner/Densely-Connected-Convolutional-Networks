from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

channels = 1
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

images = X_train.reshape(-1, 28, 28, 1).astype('float64')
images_means = []
images_stds = []
for i in range(channels):
    mean = np.mean(images[:, :, :, i])
    std = np.std(images[:, :, :, i])
    images_means.append(mean)
    images_stds.append(std)

for i in range(channels):
    images[:, :, :, i] = ((images[:, :, :, i] - images_means[i]) / images_stds[i])

labels = to_categorical(Y_train)

images_test = X_test.reshape(-1, 28, 28, 1).astype('float64')

images_test_means = []
images_test_stds = []
for i in range(channels):
    mean = np.mean(images_test[:, :, :, i])
    std = np.std(images_test[:, :, :, i])
    images_test_means.append(mean)
    images_test_stds.append(std)

for i in range(channels):
    images_test[:, :, :, i] = ((images_test[:, :, :, i] - images_test_means[i]) / images_test_stds[i])

labels_test = Y_test