import tensorflow
from tensorflow import keras
import numpy
import os
import sklearn
import sklearn.preprocessing
import sklearn.model_selection

X = numpy.zeros((51, 210, 3))   # 21 samples, 210 timesteps, 3 features
Y = numpy.zeros((51)) # каждому файлу исходных данных соответствует лейбл - 0 для нейтральных, -1 для отрицательных и 1 для положительных
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
i = 0

# получение данных из файлов
for filename in os.listdir('dataset\\negative'):
    X[i] = numpy.loadtxt(open("dataset\\negative\\" + filename, "rb"), delimiter=",", skiprows=1)  # собрать данные из очередного файла
    X[i] = scaler.fit_transform(X[i])  # смасштабировать данные из очередного файла
    Y[i] = -1
    i += 1

for filename in os.listdir('dataset\\neutral'):
    X[i] = numpy.loadtxt(open("dataset\\neutral\\" + filename, "rb"), delimiter=",", skiprows=1)  # собрать данные из очередного файла
    X[i] = scaler.fit_transform(X[i])  # смасштабировать данные из очередного файла
    Y[i] = 0
    i += 1

for filename in os.listdir('dataset\\positive'):
    X[i] = numpy.loadtxt(open("dataset\\positive\\" + filename, "rb"), delimiter=",", skiprows=1)  # собрать данные из очередного файла
    X[i] = scaler.fit_transform(X[i])  # смасштабировать данные из очередного файла
    Y[i] = 1
    i += 1

le = sklearn.preprocessing.OneHotEncoder()
Y = numpy.array(le.fit_transform(Y.reshape(-1, 1)).toarray())    # лейблы преобразовываются из массива чисел в массив векторов, где [1, 0, 0] обозначает -1, [0, 1, 0] обозначает 0 и [0, 0, 1] обозначает 1

print(X)
print(Y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2) #данные разделяются на тренировочные и тестировочные

#Строится модель
model = keras.models.Sequential()
model.add(keras.layers.LSTM(5, input_shape=(210, 3), activation='elu',  kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Dense(3, activation='sigmoid'))

model.summary()

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

x_train = numpy.asarray(x_train).astype(numpy.float32)

history = model.fit(x_train, y_train, epochs=100, batch_size=1,  verbose=2, validation_data=(x_test, y_test))   #модель обучается на тестовых данных

score = model.evaluate(x_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

model.save('emotional_state.h5')