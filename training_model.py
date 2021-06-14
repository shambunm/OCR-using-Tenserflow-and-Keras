# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 14:49:01 2021

@author: shambhu
"""

from tensorflow.keras.datasets import mnist
import numpy as np

def load_az_dataset(datasetPath):
    data = []
    labels = []
    for row in open(datasetPath):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    return (data, labels)

def load_mnist_dataset():
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	return (data, labels)

import matplotlib
from tensorflow.keras.applications.resnet50 import ResNet50

import build_resnet
Resnet = build_resnet.ResNet()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

EPOCHS = 50
INIT_LR = 1e-1
BS = 128


(azData, azLabels) = load_az_dataset('/content/drive/MyDrive/Colab Notebooks/sparks/A_Z Handwritten Data.csv')
(digitsData, digitsLabels) = load_mnist_dataset()

azLabels += 10
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")
data = np.expand_dims(data, axis=-1)
data /= 255.0

le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)
classTotals = labels.sum(axis=0)

classWeight = {}
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
	rotation_range=7,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

opt = SGD(lr=INIT_LR, decay=INIT_LR / (.75*EPOCHS))
model =Resnet.build(32, 32, 1, len(le.classes_), (4, 5, 6),
	(64, 64, 128, 256))
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	class_weight=classWeight,
	verbose=1)

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))


model.save("ResNet12.h5")
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('/content/drive/MyDrive/Colab Notebooks/sparks')