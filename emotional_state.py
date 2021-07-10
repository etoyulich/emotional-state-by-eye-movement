import keras
import tensorflow
from keras.models import load_model
import sys
import os
import numpy
import sklearn
import sklearn.preprocessing
import sklearn.model_selection

if len(sys.argv) == 1: # если передан только 1 аргумент
    print("Too few command line arguments\n")
elif len(sys.argv) > 2: # если передано больше 2 аргументов
    print("Too many command line arguments\n")
else: # иначе
    if not sys.argv[1].endswith('.csv'): # если входной файл имеет не .csv расширение
        print("Invalid input data. The output file must have the .csv extension. \n")
    else: # иначе
        loaded_model = load_model('emotional_state.h5') # выгрузить модель

        # нормализорвать данные
        X = numpy.zeros((1, 210, 3))
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))

        X[0] = numpy.loadtxt(open(sys.argv[1], "rb"), delimiter=",", skiprows=1)
        X[0] = scaler.fit_transform(X[0])

        # вывести результат работы модели
        predicted_emotion = loaded_model.predict(X).squeeze()
        print("\n")
        print(predicted_emotion)