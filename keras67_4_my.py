# 나를 찍어서 내가 남자인지 여자인지에 대해 

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
    batch_size=10, 
    class_mode='binary', # 앞에있는놈이 0 뒤에는 1 
    subset='validation'
)

x_pred = train_datagen.flow_from_directory(
    './image/human/taeuk1/',
    target_size = (150,150),
    batch_size= 14,
    class_mode='binary', 
)
# predict_generater
# xy_real = train_datagen.flow_from_directory( 
#     './image/human/taeuk',
#     target_size=(150,150),
#     batch_size=1, 
#     class_mode='binary', # 앞에있는놈이 0 뒤에는 1 
#     subset='validation'
# )

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
# loss, acc :  loss, acc :  [4.355147361755371, 0.4755513072013855]
result = model.predict_generator(x_pred,verbose=True)
np.where(result < 0.5, '남자', '여자')
print(result)
print("남자일 확률은",result*100,"%입니다.")


'''
class_names = ['female','male']

import matplotlib.pyplot as plt
from PIL import Image

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = Image.open(img).convert('RGB')
    img.save(img)

    plt.imshow(img, cmap = 'gray')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        '{} {:2.0f}% ({})'.format(class_names[predicted_label],
         100*np.max(predictions_array),
         class_names[true_label]),
         color=color)

def plot_value_array(i, predictions_array,true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.gird(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np. argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i,result, xy_train, xy_val)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, result, xy_val)
plt.show()
'''