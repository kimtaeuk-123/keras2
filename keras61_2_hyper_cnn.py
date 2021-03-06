# cnn 으로 수정
# 파라미터 변경 
# 변수 : 노드의 갯수 
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Flatten
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터 / 전처리 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.

#2. 모델

def build_model(drop=0.5, optimizer='adam', node=100, ):
    inputs = Input(shape=(28,28), name='input')
    x = Conv1D(node, 3, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv1D(node, 3, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    node = [100, 200, 300]
    batchs = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    return{'batch_size' : batchs, 'optimizer': optimizers, 'drop':dropout, 'node':node}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  #머신러닝이 케라스보다 더 먼저 나와서 랩핑을 해줘야 한다
model2 = KerasClassifier(build_fn=build_model, verbose=1)



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2, hyperparameters, cv=3)  #cv cross validation
search = GridSearchCV(model2, hyperparameters, cv=3)  #cv cross validation

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # {'optimizer': 'rmsprop', 'node': 200, 'drop': 0.1, 'batch_size': 30}
print(search.best_estimator_)
print(search.best_score_) #0.972683310508728
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc ) #최종 스코어 :  0.9696999788284302


# gridsearch  로 한거 
# {'batch_size': 30, 'drop': 0.1, 'node': 300, 'optimizer': 'rmsprop'}
# 0.9750166734059652
# 최종 스코어 :  0.9793999791145325