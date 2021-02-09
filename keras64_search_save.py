# 가중치 저장할것
# 1. model.save() 쓸것 
# 2. pickle 쓸것

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터 / 전처리 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batchs = [10,20,30]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return{'batch_size' : batchs, 'optimizer': optimizers, 'drop':dropout}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  #머신러닝이 케라스보다 더 먼저 나와서 랩핑을 해줘야 한다
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)  #cv cross validation
# search = GridSearchCV(model2, hyperparameters, cv=3)  #cv cross validation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)


search.fit(x_train, y_train, verbose=1, epochs=1, validation_split=0.2, callbacks=[early_stopping,reduce_lr])

print(search.best_params_) #{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 30}
# print(search.best_estimator_)
print(search.best_score_) #0.9514166712760925
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc ) #최종 스코어 :  0.958899974822998

import pickle
pickle.dump(search.best_params_, open('./data/hyper_save/m61_pickle.data','wb'))
print('저장완료')
print('================== pickle 불러오기 ================')

model3 = pickle.load(open('./data/hyper_save/m61_pickle.data','rb'))
print('불러왔다!')
