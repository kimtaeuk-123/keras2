# 실습
# cifar10을 flow로 구성해서 완성 
# ImageDataGenerator / fit_generator -> npy 저장 

import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# dataset = cifar10()
# x = dataset.data
# y = dataset.target


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# train_test_split 다름 주의 !

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(50000, 1)
print(y_train.shape) #(10000, 32, 32, 3)
print(y_test.shape) #(10000, 1)

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255.  
x_test = x_test.reshape(10000,32,32,3)/255.  


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_test.shape) #(10000, 1)
print(y_train.shape) #(10000, 32, 32, 3)



train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, #수평방향뒤집기 
    vertical_flip=True, #수직방향뒤집기 
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest', # 빈자리를 채워준다 
    validation_split=0.4
)

# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle= True, random_state=77)

train_datagen.fit(x_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))
    
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))
    
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.2, batch_size=20, epochs=100, callbacks=[reduce_lr,early_stopping])


loss = model.evaluate(x_test, y_test)
print('loss, acc : ', loss )

# loss, acc :  [0.5301111340522766, 0.8363000154495239]