import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc


IMG_SIZE = (80, 80)
channels = 1
an_path = r'C:\Users\u9133908\Documents\GitHub\do\..py\GenDocks-main\aa'

an_dict = {}
for ani in os.listdir(an_path):
    an_dict[ani] = len(os.listdir(os.path.join(an_path, ani)))

an_dict = caer.sort_dict(an_dict, descending=True)
print(an_dict)

rp = []
for an in an_dict:
    rp.append(an[0])
print(rp)


train = caer.preprocess_from_dir(an_path, rp, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

featset, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
# Normalize featset ==> (0,1)
from tensorflow.keras.utils import to_categorical
featset = caer.normalize(featset)
labels = to_categorical(labels, len(rp))

x_train, x_val, y_train, y_val = caer.train_val_split(featset, labels, val_ratio=.2)

BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

model = canaro.models.createDefaultModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dims=len(rp), loss='binary_crossentropy', decay=1e-6, learning_rate=0.001, momentum=0.9, nesterov=True)
model.summary()
