import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, #수평방향뒤집기 
    vertical_flip=True, #수직방향뒤집기 
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' # 빈자리를 채워준다 
)

test_datagen = ImageDataGenerator(rescale=1./255)
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
# 테스트는 왜 리스케일만 할까??
# 시험문제는 건들 필요가 없기 때문 
# 원래 0~255 사이 값인데 /255 해줘서 0~1 값이다 


# test ?
# ad 폴더 = 치매 
# x = (80장x150,150,1)(흑백=1, 칼라=3) (0~1사이 값이 들어가있음 /255 해줬기 때문)
# y = (80장,) (값 0이 들어가 있음)

# normar 
# y = (80,) (값 1이 들어가 있음)

# 폴더 채로
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# train_generater
xy_train = train_datagen.flow_from_directory( 
    './image/brain/train',
    target_size=(150,150),
    batch_size=5, 
    class_mode='binary' 
    #( 80, 150, 150, 1)
    # 앞에있는놈이 0 뒤에는 1 
)

# test_generater
xy_test = test_datagen.flow_from_directory( 
    './image/brain/train',
    target_size=(150,150),
    batch_size=5, 
    class_mode='binary' # 앞에있는놈이 0 뒤에는 1 
)

model = Sequential()
# model.add(Conv2D(32,(3,3), input_shape=(150,150,3)))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit_generator(
    xy_train, steps_per_epoch=31, epochs=100,
    validation_data=xy_test, validation_steps=4
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    # steps_per_epoch 33 일땐 안되고 31일땐 된다. => 160개 / 5 => 32 
    # 32보다 적거나 많을땐 손실을 본다 (33을 넣어주면 안돌아가고 31을 넣어주면 32보다 조금밖에안돌아가니 손해)
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    )

    
# 시각화 


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 빈자리에 설명(범주?)
plt.show()


'''
plt.figure(figsize=(10,6)) #면적잡기, 판깔기
plt.subplot(2,1,1) # 2행 1열 중 첫번째
#이미지 2개 뽑겠다. -> (2,1) 즉 2행 1열짜리 하나 만들겠다?
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('cost loss')
# plt.title('손실비용') 한글 안됨 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2,1,2) #2행 2열 중 두번째 
plt.plot(hist.history['acc'], marker='.', c='red', label='acc') #accuracy 면 accuracy로 써야한다 
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid() #모눈종이 격자 

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

'''
# print('acc :', acc[-1])  #( acc 의 마지막 )
# print('val_acc :', val_acc[:-1]) #( epochs = 30 개니가 30개 나오고 전체가 나온다 .)