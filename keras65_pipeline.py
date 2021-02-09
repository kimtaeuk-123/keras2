import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터 / 전처리 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')
x_test = x_test.reshape(10000, 28*28).astype('float32')

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
    batchs = [10,20]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return{'clf__batch_size' : batchs, 'clf__optimizer': optimizers, 'clf__drop':dropout}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  #머신러닝이 케라스보다 더 먼저 나와서 랩핑을 해줘야 한다


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC

model2 = KerasClassifier(build_fn=build_model, epochs=1, batch_size=32, verbose=1)
pipe = Pipeline([("scaler",MinMaxScaler()), ('clf',model2)])
kfold = KFold(n_splits=3, random_state=42)
search = GridSearchCV(pipe, hyperparameters, cv=kfold)

search.fit(x_train, y_train)
acc = search.score(x_test, y_test)
results = search.score(x_test, y_test)
print('results : ', results)
print("최종 스코어 :", acc)
print(search.best_params_) 
print(search.best_score_)
# results :  0.963100016117096
# 최종 스코어 : 0.963100016117096
# {'clf__batch_size': 20, 'clf__drop': 0.1, 'clf__optimizer': 'adam'}
# 0.9564666748046875






# print(search.best_estimator_) # 모든 파라미터들 근데 케라스 파라미터 인식을 못해