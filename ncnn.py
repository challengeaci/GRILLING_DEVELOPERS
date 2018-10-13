from keras.layers import Dense , Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.models import Sequential
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.optimizers import SGD
from keras.models import load_model
import glob2
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from keras.backend.tensorflow_backend import set_session
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import keras

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

def denoise(I):
	sigma = 0.08
	sigma_est = np.mean(estimate_sigma(I, multichannel=False))
	patch_kw = dict(patch_size=5,     
                patch_distance=6,  
                multichannel=False)
	I = denoise_nl_means(I, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
	return I

def snakes(image):
	l=np.zeros(shape=(256,256))
	def store_evolution_in(lst):
		def _store(x):
			lst.append(np.copy(x))
		return _store
		max=np.amax(image)
		eimage=(image*255/max)**5
		init_ls=checkerboard_level_set(image.shape,6)
		evolution=[]
		callback=store_evolution_in(evolution)
		ls=morphological_chan_vese(eimage,35,init_level_set=init_ls,smoothing=3,iter_callback=callback)
		l=ls
	return l

i=0
path = 'C:\\Users\\Kashyap YV\\PROJECT\\training\\tumor./*'	
files = glob2.glob(path)
print(len(files))

I=np.zeros(shape=(256,256,3,184))
for name in files:
	img=pydicom.dcmread(files[i]).pixel_array
	print(files[i])
	img=denoise(img)
	I[:,:,0,i]=img
	I[:,:,1,i]=I[:,:,0,i]
	I[:,:,2,i]=I[:,:,0,i]
	i=i+1
o_t=np.ones(shape=(1,i))
k=0
path = 'C:\\Users\\Kashyap YV\\PROJECT\\training\\normal./*'	
files = glob2.glob(path)
print(len(files))
for name in files:
	img=pydicom.dcmread(files[k]).pixel_array
	print(files[k])
	img=denoise(img)
	I[:,:,0,i]=img
	I[:,:,1,i]=I[:,:,0,i]
	I[:,:,2,i]=I[:,:,0,i]
	i=i+1
	k=k+1
o_n=np.zeros(shape=(1,k))
o = np.append(o_t,o_n)
#I=np.random.rand(256,256,3,1000)

I=np.swapaxes(I,0,3)
I=np.swapaxes(I,1,3)
I=np.swapaxes(I,2,3)

I=I/I.max()

f = open('I.pckl', 'wb')
pickle.dump(I, f)
f.close()

f = open('I.pckl', 'rb')
I = pickle.load(f)
f.close()

o_t=np.ones(shape=(1,92))
o_n=np.zeros(shape=(1,92))
o=np.append(o_t,o_n)
model=Sequential()

model.add(Conv2D(32,(4,4), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(4,4), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(4,4), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

'''
datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
'''
x_train,x_test,y_train,y_test = train_test_split(I,o,test_size=0.2)

#datagen.fit(I)
#model.fit_generator(datagen.flow(x_train, y_train,batch_size=10),steps_per_epoch=147,epochs=32)

model.fit(x_train,y_train,batch_size=50,epochs=10)
score = model.evaluate(x_train, y_train, batch_size=4)
print(score)
model.save('ncnn_1.h5')
xmodel = load_model('ncnn_1.h5')
prediction=xmodel.predict(x_train)
print(pd.DataFrame({'Predicted Data':prediction[0,0], 'Actual Values':y_train}))