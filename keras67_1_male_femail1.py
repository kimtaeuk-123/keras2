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
    validation_split=0.3
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
early_stopping= EarlyStopping(monitor = 'val_loss', patience = 25, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit_generator(
    xy_train,  epochs=100, 
    validation_data=xy_val, callbacks=[early_stopping, reduce_lr]
)

loss = model.evaluate(xy_train)
print('loss, acc : ', loss)


# loss, acc :  [0.6319777369499207, 0.6348684430122375]




'''
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 빈자리에 설명(범주?)
plt.show()
'''