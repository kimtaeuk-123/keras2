import numpy as np

x_train=np.load('./image/brain/npy/keras66_train_x.npy')
y_train=np.load('./image/brain/npy/keras66_train_y.npy')
x_test=np.load('./image/brain/npy/keras66_test_x.npy')
y_test=np.load('./image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=10, verbose=1)

