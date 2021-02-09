# 실습
# 남자여자구별  
# ImageDataGenerator / fit 사용해서 완성 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

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

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory( 
    './image/human',
    target_size=(150,150),
    batch_size=10, 
    class_mode='binary', 
    #(14, 150, 150, 3)
    subset='training'
    
)

# test_generater
xy_val = train_datagen.flow_from_directory( 
    './image/human',
    target_size=(150,150),
    batch_size=1, 
    class_mode='binary', # 앞에있는놈이 0 뒤에는 1 
    subset='validation'
)
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

np.save('./image/human/npy/keras66_train_x2.npy', arr=xy_train[0][0]) # x 
np.save('./image/human/npy/keras66_train_y2.npy', arr=xy_train[0][1]) # y 
np.save('./image/human/npy/keras66_test_x2.npy', arr=xy_val[0][0]) # x 
np.save('./image/human/npy/keras66_test_y2.npy', arr=xy_val[0][1]) # y 

x_train=np.load('./image/human/npy/keras66_train_x2.npy')
y_train=np.load('./image/human/npy/keras66_train_y2.npy')
x_val=np.load('./image/human/npy/keras66_test_x2.npy')
y_val=np.load('./image/human/npy/keras66_test_y2.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(150,150,3)))
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
model.add(Dense(1, activation='sigmoid'))



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stopping= EarlyStopping(monitor = 'val_loss', patience = 300, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 200, factor = 0.5, verbose = 1)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(
    x_train,y_train,  epochs=1000, 
    validation_data=(x_val,y_val), callbacks=[early_stopping, reduce_lr]
)

loss = model.evaluate(x_val,y_val)
print('loss, acc : ', loss)

# 할때마다 다르다..
# loss, acc :  [0.5951874256134033, 0.8999999761581421]

# batchsize 를 trian - 10 , val - 1 로 서로  다르게해주니
# loss, acc :   [0.16041973233222961, 1.0] ???

