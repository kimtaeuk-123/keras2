import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# train_generater
xy_train = train_datagen.flow_from_directory( 
    './image/brain/train',
    target_size=(150,150),
    batch_size=30, 
    class_mode='binary' , save_to_dir='./image/brain/brain_generator/train/'
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    # print(xy_train[0][0])  #정의된 변수 건드려줘야한다!
    # print(xy_train[0][0])  60개
    # print(xy_train[0][0])  90개
    # ...print로 건드리는만큼 배치사이즈에 따라서 나온다 

    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

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

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
print(xy_train[0][0])  #정의된 변수 건드려줘야한다!
# print(xy_train[0][0])  60개
# print(xy_train[0][0])  90개
# ...print로 건드리는만큼 배치사이즈에 따라서 나온다 

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
