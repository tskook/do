import cv2
import numpy as np
import os
from random import shuffle
from numpy.core.records import fromfile
from numpy.lib.npyio import load
from numpy.lib.type_check import imag
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


train_dir = 'C:\\Users\\u9133908\\Documents\\GitHub\\do\\..py\\GenDocks-main\\train\\redpanda'
test_dir = 'C:\\Users\\u9133908\\Documents\\GitHub\\do\\..py\\GenDocks-main\\test'

img_size = 60
lr = 1e-3
MODEL_NAME = f'redpanda-{lr}-2conv-basic.model'

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'redpanda': return [1,0]
    else: return [0,1]

def create_train_data():
    train_data = []
    for img in os.listdir(train_dir):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data

def process_test_data():
    testing_data = []
    for img in os.listdir(test_dir):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()


convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists(f'{MODEL_NAME}.meta'):
    model.load(MODEL_NAME)
    print('loaded')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size)
Y = [i[0] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size)
test_y = [i[0] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)

import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # redpanda: [1,0]
    # not: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)
    model_out = model.predict([data])[0]

    # (0.999999 is default) change number to higher for more tolerance
    if np.real(model_out)[0] < 0.999999 : str_label='not'
    else: str_label='redpanda'

    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
