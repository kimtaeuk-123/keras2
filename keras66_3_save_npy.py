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
    batch_size=200,  
    class_mode='binary' 
    #( 80, 150, 150, 1)
    # 앞에있는놈이 0 뒤에는 1 
)

# test_generater
xy_test = test_datagen.flow_from_directory( 
    './image/brain/train',
    target_size=(150,150),
    batch_size=200, 
    class_mode='binary' # 앞에있는놈이 0 뒤에는 1 
)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
print(xy_train)
# print(xy_train[0])
# print(xy_train[0][0]) # x만 나옴  
print(xy_train[0][0].shape) #(160, 150, 150, 3)
# print(xy_train[0][1]) # y만 나옴 
print(xy_train[0][1].shape) # (160,)
# batch_size 10 일때
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

np.save('./image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0]) # x 
np.save('./image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1]) # y 
np.save('./image/brain/npy/keras66_test_x.npy', arr=xy_test[0][0]) # x 
np.save('./image/brain/npy/keras66_test_y.npy', arr=xy_test[0][1]) # x 

x_train=np.load('./image/brain/npy/keras66_train_x.npy')
y_train=np.load('./image/brain/npy/keras66_train_y.npy')
x_test=np.load('./image/brain/npy/keras66_test_x.npy')
y_test=np.load('./image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)