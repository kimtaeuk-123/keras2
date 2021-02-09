import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_boston

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)


#1. 데이터 / 전처리 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape) # (404, 13)
print(x_test.shape) # (102, 13)
print(y_train.shape)

x_train = x_train.reshape(404, 13).astype('float32')/255.
x_test = x_test.reshape(102, 13).astype('float32')/255.

kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 


#2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(13), name='input')
    x = Dense(100, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(30, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(51, activation='relu', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameters():
    batchs = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return{'batch_size' : batchs, 'optimizer': optimizers, 'drop':dropout}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor  #머신러닝이 케라스보다 더 먼저 나와서 랩핑을 해줘야 한다
model2 = KerasRegressor(build_fn=build_model, verbose=1)



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)  #cv cross validation
# search = GridSearchCV(model2, hyperparameters, cv=3)  #cv cross validation

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) #{'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 50}
print(search.best_estimator_)
print(search.best_score_) #0.9588499863942465
# scores = cross_val_score(model2,x_train,y_train, cv=kfold) # cross_validation
# print(scores)
# acc = search.score(x_test, y_test)
# print('최종 mae : ', mae ) #최종 스코어 :  0.9682999849319458

#{'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 30} 고쳐도 된다 
