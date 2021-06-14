# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:00:38 2021

@author: shambhu
"""

import numpy as np
from keras.models import load_model
import cv2
import imutils
from imutils.contours import sort_contours
model = load_model('ResNet12.h5')

path = r'C:\Users\shambhu\Artificial Intelligence\sparks_shambhu\OCR\snm.jpg'
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
# cnts[10][0][0]
    # cv2.imshow("Image", edged)
    # cv2.waitKey(0)
# edged.shape
# len(cnts)

chars = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w*h>=4900):
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)
        else:
            thresh = imutils.resize(thresh, height=32)
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                 left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0))
        padded = cv2.resize(padded, (32, 32))
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        chars.append((padded, (x, y, w, h)))
len(chars)

boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
preds = model.predict(chars)
# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    start_point  = (x, y)
    end_point  = (x + w, y + h)
    color  = (255,0,255)
    thickness  = 2
    cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, thickness)
cv2.imwrite("OCR.jpg", image)
cv2.imshow("Image", image)
cv2.waitKey(0)
