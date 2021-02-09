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

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
# array = y값 5개 => battch_size 5개 이기 때문에 

# print(xy_train)
# print(xy_train[0])
# print(xy_train[0][0]) # x만 나옴  
# print(xy_train[0][0].shape) #(5, 150, 150, 3)
# print(xy_train[0][1]) # y만 나옴 
# print(xy_train[0][1].shape) # (5,)
# # batch_size 10 일때
# # 160장의 데이터를 10으로 나누면 0~15까지만 존재 
# print(xy_train[15][1].shape) # (10,)

# batch_size 를 y라벨보다 더큰 값을 넣어주면 그대로 나온다. => y값을 알려면 아예 큰값을 줘버려서 값을 알아낼수있다

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

